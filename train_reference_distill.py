"""Successor-teacher refinement with late checkpoint averaging.

Arithmetic teacher probabilities, T=1 soft CE, cosine decay, and FP64 late
averaging are used for refinement.
"""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from common import PROTOCOL, load_data, make_model, setup, sha
from evaluate import score
from train import learning_rate_at_step


@torch.no_grad()
def teacher_probabilities(teachers, inputs, temperature):
    """Equal arithmetic mixture of member probabilities, never mean logits."""
    if not teachers or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive temperature and at least one teacher required')
    probabilities = None
    for teacher in teachers:
        if teacher.training or any(p.requires_grad for p in teacher.parameters()):
            raise ValueError('Teachers must be frozen in evaluation mode')
        member = F.softmax(teacher(inputs).float() / temperature, dim=-1)
        if probabilities is None:
            probabilities = member / len(teachers)
        else:
            probabilities.add_(member, alpha=1. / len(teachers))
    return probabilities


def objective(logits, probabilities, targets, temperature):
    """Half hard CE, half T^2 soft CE; frozen entropy does not affect gradients."""
    hard = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
    soft = -(probabilities.detach() * F.log_softmax(logits.float() / temperature, -1)).sum(-1).mean()
    return .5 * hard + .5 * temperature ** 2 * soft, hard.detach(), soft.detach()


def late_steps(steps):
    if steps < 40 or steps % 40:
        raise ValueError('Steps must be a multiple of 40 and at least 40')
    return [steps * numerator // 40 for numerator in (37, 38, 39, 40)]


def accumulate_average(average, state):
    """Accumulate four floating states in FP64 before a single FP32 export."""
    if average is None:
        return {key: value.detach().cpu().double().clone() for key, value in state.items()}
    for key, value in state.items():
        average[key].add_(value.detach().cpu().double())
    return average


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--teachers', type=Path, nargs='+', required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--temperature', type=float, required=True)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    saves = late_steps(args.steps)
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        p.error('Temperature must be finite and positive')
    if args.run_dir.exists() and any(args.run_dir.iterdir()) and not args.resume:
        p.error('Use a fresh run directory or explicit --resume')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    device, _ = setup('cuda', 'fp32', 4)
    torch.manual_seed(args.seed)
    source = torch.load(args.source, map_location='cpu', weights_only=True)
    if source['protocol'] != PROTOCOL:
        raise ValueError('Wrong source protocol')
    config = dict(source['config'], cache_weight=0.)
    config.pop('neural_temperature', None)
    model, implementation_sha = make_model('student', config, device)
    model.load_state_dict(source['model'])
    teachers, teacher_info = [], []
    with torch.random.fork_rng(devices=[device.index]):
        for path in args.teachers:
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved['protocol'] != PROTOCOL:
                raise ValueError('Wrong teacher protocol')
            teacher, digest = make_model(saved['implementation'], saved['config'], device)
            teacher.load_state_dict(saved['model'])
            teacher.eval().requires_grad_(False)
            teachers.append(teacher)
            teacher_info.append(dict(path=str(path), sha256=sha(path),
                                     implementation=saved['implementation'], implementation_sha256=digest,
                                     student_targets=saved['train_tokens'],
                                     inherited_teacher=saved.get('distillation_teacher')))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.00012, weight_decay=.1, betas=(.9, .999))
    rng = torch.Generator().manual_seed(args.seed)
    data = load_data()
    tokens = data['train'][0].to(device)
    recipe = dict(seed=args.seed, steps=args.steps, batch_size=32, context=256,
                  peak_lr=.00012, warmup=50, min_lr_ratio=.1, schedule='cosine',
                  distill_weight=.5, temperature=args.temperature, weight_decay=.1,
                  optimizer_restart=True, average_steps=saves, average_accumulation='fp64',
                  source_sha256=sha(args.source), teachers=teacher_info)
    state_path = args.run_dir / 'training-state.pt'
    start_step, average, averaged_steps, history, elapsed_prior = 0, None, [], [], 0.
    if args.resume:
        state = torch.load(state_path, map_location='cpu', weights_only=True)
        if state['recipe'] != recipe:
            raise ValueError('Changed resume recipe')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        rng.set_state(state['sampling_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        start_step, average = state['step'], state['average']
        averaged_steps, history, elapsed_prior = state['averaged_steps'], state['history'], state['train_seconds']
    (args.run_dir / 'recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')

    def payload(weights, step, average_steps=None):
        output = dict(protocol=PROTOCOL, implementation='student', config=config,
                      model=weights, seed=args.seed, train_tokens=source['train_tokens'] + step * 8192,
                      additional_train_targets=step * 8192, optimizer_recipe=recipe,
                      distillation_teacher=dict(members=teacher_info, mixture='equal arithmetic probabilities',
                                                temperature=args.temperature),
                      ancestry=dict(checkpoint=str(args.source), sha256=recipe['source_sha256'],
                                    source_student_targets=source['train_tokens'], optimizer_restart=True),
                      test_evaluated=False)
        if average_steps is not None:
            output['averaged_steps'] = average_steps
        return output

    def validations():
        neural = score(model, *data['validation'], device, 'fp32')
        neural.pop('window_nll_nats')
        from student_temperature_fastcache import FastTemperatureStudentGPT
        cached = FastTemperatureStudentGPT(dict(config, neural_temperature=1.05,
                                                cache_theta=13., cache_weight=.085)).to(device)
        cached.load_state_dict(model.state_dict())
        result = score(cached, *data['validation'], device, 'fp32')
        result.pop('window_nll_nats')
        return dict(neural=neural, cached=result)

    torch.cuda.synchronize()
    started, excluded = time.perf_counter(), 0.
    validation_steps = {args.steps // 3, 2 * args.steps // 3, args.steps}
    for index in range(start_step, args.steps):
        step = index + 1
        model.train()
        starts = torch.randint(len(tokens) - 257, (32,), generator=rng).to(device)
        batch = tokens[starts[:, None] + torch.arange(257, device=device)]
        inputs, targets = batch[:, :-1], batch[:, 1:]
        lr = learning_rate_at_step(index, args.steps, .00012, 50, .1, 'cosine')
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        probabilities = teacher_probabilities(teachers, inputs, args.temperature)
        loss, hard, soft = objective(model(inputs), probabilities, targets, args.temperature)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training objective')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(grad_norm):
            raise RuntimeError('Nonfinite gradient norm')
        optimizer.step()
        if step % 100 == 0:
            print(json.dumps(dict(step=step, loss=loss.item(), hard=hard.item(), soft=soft.item(), lr=lr)), flush=True)
        if step in saves:
            average = accumulate_average(average, model.state_dict())
            averaged_steps.append(step)
        if step in validation_steps or step in saves:
            weights = model.state_dict()
            torch.save(payload(weights, step), args.run_dir / f'checkpoint-step{step}.pt')
        if step in validation_steps:
            torch.cuda.synchronize()
            observed = time.perf_counter()
            with torch.random.fork_rng(devices=[device.index]):
                result = validations()
            torch.cuda.synchronize()
            excluded += time.perf_counter() - observed
            row = dict(step=step, **result)
            history.append(row)
            print(json.dumps(dict(validation=row)), flush=True)
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            sampling_rng=rng.get_state(), torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all(), recipe=recipe,
                            step=step, average=average, averaged_steps=averaged_steps, history=history,
                            train_seconds=elapsed_prior + time.perf_counter() - started - excluded), state_path)
    torch.cuda.synchronize()
    train_seconds = elapsed_prior + time.perf_counter() - started - excluded
    torch.save(payload(model.state_dict(), args.steps), args.run_dir / 'checkpoint.pt')
    if averaged_steps != saves:
        raise RuntimeError('Missing late-average state')
    weights = {key: (value / 4.).float() for key, value in average.items()}
    torch.save(payload(weights, args.steps, saves), args.run_dir / 'checkpoint-average.pt')
    model.load_state_dict(weights)
    average_validation = validations()
    if any(t.training or any(p.grad is not None for p in t.parameters()) for t in teachers):
        raise RuntimeError('Teacher acquired gradients or entered training mode')
    metrics = dict(recipe=recipe, validation_history=history, average_validation=average_validation,
                   averaged_steps=averaged_steps, train_seconds=train_seconds,
                   additional_train_targets=args.steps * 8192,
                   online_teacher_token_member_forwards=args.steps * 8192 * len(teachers),
                   cumulative_student_targets=source['train_tokens'] + args.steps * 8192,
                   checkpoint_sha256=sha(args.run_dir / 'checkpoint.pt'),
                   average_sha256=sha(args.run_dir / 'checkpoint-average.pt'),
                   implementation_sha256=implementation_sha, teacher_frozen=True, test_evaluated=False)
    (args.run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(dict(event='completed', average_validation=average_validation)), flush=True)


if __name__ == '__main__':
    main()

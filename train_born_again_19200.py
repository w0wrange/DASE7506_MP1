"""From-scratch independent teacher and student with configurable full-length recipes.

Uses teacher probability mixtures and FP64 late averaging. Supports
exact optimizer/sampling/CPU/CUDA RNG resume without changing the full schedule.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from common import PROTOCOL, device_metrics, load_data, make_model, setup, sha
from evaluate import score
from train import learning_rate_at_step
from train_reference_distill import accumulate_average, late_steps, objective, teacher_probabilities


def recipe_for(role, seed, config, teachers, steps=9600, weight_decay=.1,
               implementation='student_reference_rotary'):
    return dict(role=role, seed=seed, config=config, implementation=implementation,
                random_initialization=True, source_checkpoint=None,
                steps=steps, batch_size=32, context=256, peak_lr=.001, warmup=100,
                schedule='cosine', min_lr_ratio=.1, weight_decay=weight_decay, betas=[.9, .999],
                grad_clip=1., teacher_weight=.5 if role == 'student' else 0., temperature=1.,
                teachers=teachers, average_steps=late_steps(steps), average_accumulation='fp64')


def validate(model, data, device, role):
    neural = score(model, *data['validation'], device, 'fp32')
    neural.pop('window_nll_nats')
    result = dict(neural=neural)
    if role == 'student':
        previous = model.cache_weight, model.cache_theta, model.neural_temperature
        try:
            model.cache_weight, model.cache_theta, model.neural_temperature = .085, 13., 1.05
            cached = score(model, *data['validation'], device, 'fp32')
            cached.pop('window_nll_nats')
            result['cached'] = cached
        finally:
            model.cache_weight, model.cache_theta, model.neural_temperature = previous
    return result


def main():
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['teacher', 'student'], required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--implementation', default='student_reference_rotary')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--teachers', type=Path, nargs='*', default=[])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=9600)
    parser.add_argument('--weight-decay', type=float, default=.1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if args.stop_after is None:
        args.stop_after = args.steps
    if (args.role == 'teacher' and args.teachers) or (args.role == 'student' and len(args.teachers) != 2):
        parser.error('Teacher learns only hard labels; student requires exactly two teacher checkpoints')
    if args.steps < 100 or args.steps % 40 or not 0 < args.stop_after <= args.steps:
        parser.error('Steps must be a multiple of 40, at least 100; stop boundary must fit full schedule')
    if not 0 <= args.weight_decay < 1:
        parser.error('Weight decay must be in [0, 1)')
    if args.run_dir.exists() and any(args.run_dir.iterdir()) and not args.resume:
        parser.error('Use a fresh directory or explicit --resume')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    device, _ = setup('cuda', 'fp32', 4)
    torch.manual_seed(args.seed)
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    model.train()
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
            teacher_info.append(dict(checkpoint=str(path), sha256=sha(path),
                                     implementation=saved['implementation'], implementation_sha256=digest,
                                     student_targets=saved['train_tokens'], seed=saved.get('seed'),
                                     inherited_teacher=saved.get('distillation_teacher')))
    recipe = recipe_for(args.role, args.seed, config, teacher_info, args.steps,
                        args.weight_decay, args.implementation)
    recipe['config_sha256'] = sha(args.config)
    if args.implementation != 'student_reference_rotary':
        recipe['implementation_sha256'] = implementation_sha
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .999),
                                  weight_decay=args.weight_decay)
    sampling = torch.Generator().manual_seed(args.seed)
    data = load_data()
    tokens = data['train'][0].to(device)
    step_start, average, averaged_steps, history, prior_seconds = 0, None, [], [], 0.
    state_path = args.run_dir / 'training-state.pt'
    initial_path = args.run_dir / 'checkpoint-initial.pt'
    if args.resume:
        saved_state = torch.load(state_path, map_location='cpu', weights_only=True)
        if saved_state['recipe'] != recipe:
            raise ValueError('Resume source, teacher or recipe changed')
        model.load_state_dict(saved_state['model'])
        optimizer.load_state_dict(saved_state['optimizer'])
        sampling.set_state(saved_state['sampling_rng'])
        torch.set_rng_state(saved_state['torch_rng'])
        torch.cuda.set_rng_state_all(saved_state['cuda_rng'])
        step_start, average = saved_state['step'], saved_state['average']
        averaged_steps, history, prior_seconds = (saved_state['averaged_steps'], saved_state['history'],
                                                  saved_state['update_seconds_including_in_loop_io'])
    else:
        torch.save(dict(protocol=PROTOCOL, implementation=args.implementation, config=config,
                        model=model.state_dict(), seed=args.seed, train_tokens=0,
                        random_initialization=True), initial_path)
    if step_start >= args.stop_after:
        raise ValueError('Resume boundary must extend past saved step')
    (args.run_dir / 'recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')

    def payload(weights, step, averaged=None):
        saved = dict(protocol=PROTOCOL, implementation=args.implementation, config=config,
                     model=weights, seed=args.seed, step=step, train_tokens=step * 8192,
                     optimizer_recipe=recipe, random_initialization=True, test_evaluated=False,
                     initialization_checkpoint_sha256=sha(initial_path))
        if args.role == 'student':
            saved['distillation_teacher'] = dict(members=teacher_info, mixture='equal arithmetic probabilities',
                                                temperature=1., teacher_weight=.5)
        if averaged is not None:
            saved['averaged_steps'] = averaged
        return saved

    torch.cuda.synchronize()
    started, excluded = time.perf_counter(), 0.
    saves = late_steps(args.steps)
    for index in range(step_start, args.stop_after):
        step = index + 1
        model.train()
        starts = torch.randint(len(tokens) - 257, (32,), generator=sampling).to(device)
        batch = tokens[starts[:, None] + torch.arange(257, device=device)]
        inputs, targets = batch[:, :-1], batch[:, 1:]
        lr = learning_rate_at_step(index, args.steps, .001, 100, .1, 'cosine')
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        if teachers:
            q = teacher_probabilities(teachers, inputs, 1.)
            loss, hard, soft = objective(model(inputs), q, targets, 1.)
        else:
            loss = F.cross_entropy(model(inputs).float().flatten(0, 1), targets.flatten())
            hard, soft = loss.detach(), loss.detach().new_zeros(())
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training objective')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(grad_norm):
            raise RuntimeError('Nonfinite gradient norm')
        optimizer.step()
        if step % 100 == 0:
            print(json.dumps(dict(role=args.role, step=step, hard=hard.item(), soft=soft.item(),
                                  loss=loss.item(), lr=lr)), flush=True)
        if step in saves:
            average = accumulate_average(average, model.state_dict())
            averaged_steps.append(step)
        if step % 800 == 0 or step in saves or step == args.stop_after:
            torch.save(payload(model.state_dict(), step), args.run_dir / f'checkpoint-step{step}.pt')
            torch.cuda.synchronize()
            observed = time.perf_counter()
            with torch.random.fork_rng(devices=[device.index]):
                result = validate(model, data, device, args.role)
            torch.cuda.synchronize()
            excluded += time.perf_counter() - observed
            row = dict(step=step, **result)
            history.append(row)
            (args.run_dir / 'status.json').write_text(json.dumps(dict(status='running', **row)) + '\n')
            with (args.run_dir / 'curve.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print(json.dumps(dict(validation=row)), flush=True)
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), recipe=recipe,
                            step=step, sampling_rng=sampling.get_state(), torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all(), average=average, averaged_steps=averaged_steps,
                            history=history, update_seconds_including_in_loop_io=(
                                prior_seconds + time.perf_counter() - started - excluded)), state_path)
    torch.cuda.synchronize()
    update_seconds = prior_seconds + time.perf_counter() - started - excluded
    if args.stop_after < args.steps:
        print(json.dumps(dict(event='stopped_at_declared_boundary', step=args.stop_after,
                              full_schedule=args.steps)), flush=True)
        return
    torch.save(payload(model.state_dict(), args.steps), args.run_dir / 'checkpoint.pt')
    if averaged_steps != saves:
        raise RuntimeError('Missing late averaging state')
    averaged_weights = {key: (value / 4).float() for key, value in average.items()}
    torch.save(payload(averaged_weights, args.steps, saves), args.run_dir / 'checkpoint-average.pt')
    model.load_state_dict(averaged_weights)
    average_validation = validate(model, data, device, args.role)
    if any(t.training or any(p.grad is not None or p.requires_grad for p in t.parameters()) for t in teachers):
        raise RuntimeError('Teacher was not frozen')
    metrics = dict(recipe=recipe, parameters=sum(p.numel() for p in model.parameters()),
                   validation_history=history, average_validation=average_validation, averaged_steps=averaged_steps,
                   additional_train_targets=args.steps * 8192,
                   online_teacher_token_member_forwards=args.steps * 8192 * len(teachers),
                   update_seconds_including_in_loop_io=update_seconds, validation_seconds=excluded,
                   process_seconds_this_invocation=time.perf_counter() - process_started,
                   checkpoint_sha256=sha(args.run_dir / 'checkpoint.pt'),
                   average_sha256=sha(args.run_dir / 'checkpoint-average.pt'),
                   initialization_sha256=sha(initial_path), teacher_frozen=True,
                   random_initialization=True, test_evaluated=False, **device_metrics(device))
    (args.run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    (args.run_dir / 'status.json').write_text(json.dumps(dict(status='complete', step=args.steps,
                                                           average_validation=average_validation)) + '\n')
    print(json.dumps(dict(event='completed', role=args.role, average_validation=average_validation)), flush=True)


if __name__ == '__main__':
    main()

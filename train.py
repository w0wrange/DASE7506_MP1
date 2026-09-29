"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def learning_rate_at_step(step, steps, peak_lr, warmup_steps, min_lr_ratio,
                          schedule='cosine', decay_start=800):
    """Warm up linearly while applying the original full-run cosine schedule."""
    warmup = 1. if warmup_steps == 0 else min(1., (step + 1) / warmup_steps)
    if schedule == 'wsd':
        progress = max(0., (step + 1 - decay_start) / max(1, steps - decay_start))
        return peak_lr * warmup * (1. - (1. - min_lr_ratio) * min(progress, 1.))
    if schedule != 'cosine':
        raise ValueError('Unknown learning-rate schedule.')
    cosine = min_lr_ratio + (1. - min_lr_ratio) * .5 * (
        1. + math.cos(math.pi * step / steps)
    )
    return peak_lr * warmup * cosine


def embedding_lr_groups(groups, model, multiplier):
    """Split the input token weight out of each decay group, once when tied."""
    if multiplier == 1.:
        return groups
    result = []
    for group in groups:
        for is_token in (False, True):
            params = [p for p in group['params'] if (p is model.token.weight) == is_token]
            if params:
                result.append({**group, 'params': params,
                               'lr_scale': multiplier if is_token else 1.})
    return result


def average_state(average, state, count):
    """Return the running arithmetic mean of floating checkpoint tensors."""
    if average is None:
        return {name: value.detach().cpu().clone() for name, value in state.items()}
    for name, value in state.items():
        value = value.detach().cpu()
        if value.is_floating_point():
            average[name].add_((value - average[name]) / (count + 1))
        else:
            average[name].copy_(value)
    return average


def optimizer_groups(model, weight_decay, decay_mode):
    """Build disjoint AdamW groups and return serializable group statistics.

    ``all`` preserves the supplied recipe. ``matrix`` applies decay to learned
    matrices while excluding biases and one-dimensional normalization scales.
    Shared parameters (the tied token/head weight) are included exactly once by
    ``named_parameters``.
    """
    named = [(name, parameter) for name, parameter in model.named_parameters()
             if parameter.requires_grad]
    if decay_mode == 'all':
        groups = [{'params': [parameter for _, parameter in named],
                   'weight_decay': weight_decay}]
        stats = {'mode': decay_mode, 'weight_decay': weight_decay,
                 'decayed_tensors': len(named),
                 'decayed_parameters': sum(parameter.numel() for _, parameter in named),
                 'nondecayed_tensors': 0, 'nondecayed_parameters': 0}
        return groups, stats
    if decay_mode != 'matrix':
        raise ValueError(f'Unknown decay mode: {decay_mode!r}')
    decay = [(name, parameter) for name, parameter in named if parameter.ndim >= 2]
    no_decay = [(name, parameter) for name, parameter in named if parameter.ndim < 2]
    groups = [
        {'params': [parameter for _, parameter in decay], 'weight_decay': weight_decay},
        {'params': [parameter for _, parameter in no_decay], 'weight_decay': 0.},
    ]
    grouped_ids = [id(parameter) for group in groups for parameter in group['params']]
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != {id(p) for _, p in named}:
        raise RuntimeError('Optimizer parameter groups must be complete and disjoint.')
    stats = {'mode': decay_mode, 'weight_decay': weight_decay,
             'decayed_tensors': len(decay),
             'decayed_parameters': sum(parameter.numel() for _, parameter in decay),
             'nondecayed_tensors': len(no_decay),
             'nondecayed_parameters': sum(parameter.numel() for _, parameter in no_decay),
             'decayed_names': [name for name, _ in decay],
             'nondecayed_names': [name for name, _ in no_decay]}
    return groups, stats


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--warmup-steps', type=int, default=100)
    p.add_argument('--min-lr-ratio', type=float, default=.1)
    p.add_argument('--beta1', type=float, default=.9)
    p.add_argument('--beta2', type=float, default=.999)
    p.add_argument('--grad-clip', type=float, default=1.)
    p.add_argument('--embedding-lr-multiplier', type=float, default=1.)
    p.add_argument('--schedule', choices=['cosine', 'wsd'], default='cosine')
    p.add_argument('--decay-start', type=int, default=800)
    p.add_argument('--average-start', type=int, default=0,
                   help='First step to average; zero disables averaging.')
    p.add_argument('--average-every', type=int, default=50)
    p.add_argument('--weight-decay', type=float, default=.1)
    p.add_argument('--decay-mode', choices=['all','matrix'], default='all',
                   help='all: supplied recipe; matrix: exclude bias/norm vectors from AdamW decay.')
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.weight_decay < 0:
        p.error('Batch size and step count must be positive; weight decay must be nonnegative.')
    if args.learning_rate <= 0 or args.warmup_steps < 0:
        p.error('Learning rate must be positive and warmup steps must be nonnegative.')
    if not 0 <= args.min_lr_ratio <= 1:
        p.error('Minimum learning-rate ratio must be between zero and one.')
    if not 0 <= args.beta1 < 1 or not 0 <= args.beta2 < 1:
        p.error('AdamW beta values must be in [0, 1).')
    if args.grad_clip < 0:
        p.error('Gradient clipping must be nonnegative; use zero to disable it.')
    if not math.isfinite(args.embedding_lr_multiplier) or args.embedding_lr_multiplier <= 0:
        p.error('Embedding LR multiplier must be finite and positive.')
    if args.schedule == 'wsd' and not args.warmup_steps <= args.decay_start < args.steps:
        p.error('WSD requires warmup_steps <= decay_start < steps.')
    if not 0 <= args.average_start <= args.steps or args.average_every < 1:
        p.error('Invalid averaging start or interval.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    parameter_groups, optimizer_stats = optimizer_groups(
        model, args.weight_decay, args.decay_mode)
    parameter_groups = embedding_lr_groups(parameter_groups, model, args.embedding_lr_multiplier)
    optimizer_recipe = optimizer_stats | {
        'learning_rate': args.learning_rate,
        'warmup_steps': args.warmup_steps,
        'min_lr_ratio': args.min_lr_ratio,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'grad_clip': args.grad_clip,
        'embedding_lr_multiplier': args.embedding_lr_multiplier,
        'schedule': args.schedule,
        'decay_start': args.decay_start if args.schedule == 'wsd' else None,
    }
    optimizer = torch.optim.AdamW(
        parameter_groups, lr=args.learning_rate, betas=(args.beta1, args.beta2),
        weight_decay=0.)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    averaged = None
    averaged_steps = []
    for step in range(args.steps):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        learning_rate = learning_rate_at_step(
            step, args.steps, args.learning_rate, args.warmup_steps,
            args.min_lr_ratio, args.schedule, args.decay_start)
        for group in optimizer.param_groups:
            group['lr'] = learning_rate * group.get('lr_scale', 1.)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(),batch[:,1:].flatten())
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if (args.average_start and step + 1 >= args.average_start
                and ((step + 1 - args.average_start) % args.average_every == 0 or step + 1 == args.steps)):
            averaged = average_state(averaged, model.state_dict(), len(averaged_steps))
            averaged_steps.append(step + 1)
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            intermediate = score(model,*data['validation'],device,'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':step+1,**intermediate})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    validation = score(model,*data['validation'],device,'fp32')
    validation.pop('window_nll_nats')
    checkpoint = args.run_dir/'checkpoint.pt'
    torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                'model':model.cpu().state_dict(),'seed':args.seed,
                'train_tokens':args.steps*args.batch_size*256,
                'optimizer_recipe':optimizer_recipe},checkpoint)
    average_metadata = None
    if averaged is not None:
        average_path = args.run_dir / 'checkpoint-average.pt'
        torch.save({'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config,
                    'model': averaged, 'seed': args.seed,
                    'train_tokens': args.steps * args.batch_size * 256,
                    'optimizer_recipe': optimizer_recipe, 'averaged_steps': averaged_steps,
                    'ancestry': 'same trajectory as checkpoint.pt; no additional targets'}, average_path)
        average_metadata = {'steps': averaged_steps, 'checkpoint_sha256': sha(average_path)}
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':args.steps*args.batch_size*256,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'optimizer_recipe':optimizer_recipe,
              'weight_average': average_metadata,
              'trainer_sha256': sha(Path(__file__)),
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()

"""Train two successive born-again rounds and a matched direct control.

The existing best model's two teachers are the starting pair. Two independent
new-architecture students learn from them; a final student learns from that
new pair. A direct student with the same final seed measures the effect of the
extra generation. Every new run starts from random weights, follows the same
19,200-step AdamW recipe, and can resume at checkpoints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = 'student_qknorm_rotary'
STAGES = (
    ('g1-student-s29', 29, ('base-1', 'base-2')),
    ('g1-student-s31', 31, ('base-1', 'base-2')),
    ('g2-student-s37', 37, ('g1-student-s29', 'g1-student-s31')),
    ('direct-student-s37', 37, ('base-1', 'base-2')),
)


def parent_path(parent, run_root, base_teachers, record=True):
    if parent == 'base-1':
        return base_teachers[0]
    if parent == 'base-2':
        return base_teachers[1]
    folder = run_root / parent
    metrics_path = folder / 'metrics.json'
    if not metrics_path.exists():
        return folder / 'checkpoint-average.pt'  # Provisional dry-run path.
    metrics = json.loads(metrics_path.read_text())
    candidates = [(row['neural']['bpb'], folder / f"checkpoint-step{row['step']}.pt")
                  for row in metrics['validation_history']]
    candidates.append((metrics['average_validation']['neural']['bpb'],
                       folder / 'checkpoint-average.pt'))
    best_bpb, best_path = min(candidates, key=lambda row: row[0])
    selection_path = folder / 'teacher-selection.json'
    selected = dict(checkpoint=str(best_path), neural_validation_bpb=best_bpb,
                    checkpoint_sha256=sha256(best_path), selection_split='validation')
    if selection_path.exists() and json.loads(selection_path.read_text()) != selected:
        raise RuntimeError(f'Teacher selection changed: {selection_path}')
    if record and not selection_path.exists():
        selection_path.write_text(json.dumps(selected, indent=2) + '\n')
    return best_path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(stage, config_path, run_root, base_teachers, resume=False, dry_run=False):
    name, seed, parents = stage
    parts = [sys.executable, str(ROOT / 'train_born_again_19200.py'),
             '--role', 'student', '--implementation', IMPLEMENTATION,
             '--config', str(config_path), '--seed', str(seed),
             '--steps', '19200', '--weight-decay', '0.05',
             '--run-dir', str(run_root / name)]
    parts += ['--teachers', *(str(parent_path(p, run_root, base_teachers,
                                           record=not dry_run).relative_to(ROOT)) for p in parents)]
    if resume:
        parts.append('--resume')
    return parts


def complete(stage, config, config_path, run_root, base_teachers):
    name, seed, parents = stage
    folder = run_root / name
    paths = (folder / 'status.json', folder / 'metrics.json', folder / 'checkpoint-average.pt')
    if not all(p.exists() for p in paths):
        return False
    status = json.loads(paths[0].read_text())
    recipe = json.loads(paths[1].read_text())['recipe']
    if (status.get('status') != 'complete' or status.get('step') != 19200
            or recipe['role'] != 'student' or recipe['seed'] != seed
            or recipe['config'] != config or recipe['implementation'] != IMPLEMENTATION
            or recipe['steps'] != 19200 or recipe['weight_decay'] != .05
            or recipe.get('config_sha256') != sha256(config_path)
            or recipe.get('implementation_sha256') != sha256(ROOT / f'{IMPLEMENTATION}.py')
            or len(recipe['teachers']) != len(parents)):
        raise RuntimeError(f'Completed run has a different recipe: {folder}')
    for teacher, parent in zip(recipe['teachers'], parents):
        path = parent_path(parent, run_root, base_teachers)
        if (Path(teacher['checkpoint']).resolve() != path
                or teacher['sha256'] != sha256(path)):
            raise RuntimeError(f'Completed run has a different teacher: {folder}')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--base-teachers', type=Path, nargs=2, default=[
        ROOT / 'runs/independent-teacher-9600-s71/checkpoint-average.pt',
        ROOT / 'runs/extend-8400-fastcache-s29/checkpoint.pt'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config_path = args.config.resolve()
    run_root = args.run_root.resolve()
    base_teachers = tuple(p.resolve() for p in args.base_teachers)
    config = json.loads(config_path.read_text())
    if (config.get('heads') != 6 or config.get('width') % 12 != 0
            or config.get('depth') != 6 or config.get('ffn_hidden') != 688
            or config.get('qk_norm') is not True
            or config.get('residual_dropout') != .05 or config.get('context') != 256):
        parser.error('Expected a six-head QK-Norm config with the unchanged FFN 688')
    if args.dry_run:
        for stage in STAGES:
            print(json.dumps(dict(stage=stage[0],
                                  command=command(stage, config_path, run_root,
                                                  base_teachers, dry_run=True))))
        return
    if not all(path.is_file() for path in base_teachers):
        parser.error('Both existing base teacher checkpoints must exist')
    for stage in STAGES:
        name = stage[0]
        if complete(stage, config, config_path, run_root, base_teachers):
            print(json.dumps(dict(stage=name, status='reused')), flush=True)
            continue
        folder = run_root / name
        resume = (folder / 'training-state.pt').exists()
        if folder.exists() and any(folder.iterdir()) and not resume:
            raise RuntimeError(f'Nonempty incomplete run without resume state: {folder}')
        print(json.dumps(dict(stage=name, status='resuming' if resume else 'starting')), flush=True)
        subprocess.run(command(stage, config_path, run_root, base_teachers, resume),
                       cwd=ROOT, check=True)
        if not complete(stage, config, config_path, run_root, base_teachers):
            raise RuntimeError(f'Training ended without a complete result: {folder}')
    final_checkpoint = run_root / 'g2-student-s37' / 'checkpoint-average.pt'
    direct_checkpoint = run_root / 'direct-student-s37' / 'checkpoint-average.pt'
    print(json.dumps(dict(status='complete', born_again=str(final_checkpoint),
                          direct_control=str(direct_checkpoint))), flush=True)


if __name__ == '__main__':
    main()

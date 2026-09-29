"""Validate a faster equivalent inference implementation and freeze selection."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from common import sha  # noqa: E402
from finalize_born_again_19200 import source_checkpoint  # noqa: E402
import resource_bracket as suite  # noqa: E402

RUN_ROOT = ROOT / 'runs/born-again-qknorm348-19200'
REPORT = ROOT / 'reports/born-again-qknorm348-19200'
IMPLEMENTATION = 'student_qknorm_rotary_inference'


def export(source, destination, settings):
    saved = torch.load(source, map_location='cpu', weights_only=True)
    expected = dict(saved)
    expected['implementation'] = IMPLEMENTATION
    expected['config'] = dict(saved['config'], **settings)
    expected['inference_export'] = dict(source=str(source.relative_to(ROOT)),
                                        source_sha256=sha(source),
                                        training_implementation=saved['implementation'],
                                        additional_train_targets=0,
                                        test_evaluated=False)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(expected, destination)
    actual = torch.load(destination, map_location='cpu', weights_only=True)
    for key in ('implementation', 'config', 'inference_export'):
        if actual[key] != expected[key]:
            raise RuntimeError(f'Changed exported {key}: {destination}')
    if (actual['model'].keys() != saved['model'].keys()
            or any(not torch.equal(tensor, actual['model'][key])
                   for key, tensor in saved['model'].items())):
        raise RuntimeError(f'Changed model weights: {destination}')
    return destination


def asset_bytes(checkpoint):
    support = ('student_qknorm_rotary_inference.py', 'student_qknorm_rotary.py',
               'common.py', 'evaluate.py', 'requirements.txt',
               'data/tokenizer.json', 'data/manifest.json')
    return checkpoint.stat().st_size + sum((ROOT / name).stat().st_size for name in support)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=RUN_ROOT)
    parser.add_argument('--report', type=Path, default=REPORT)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    report = args.report.resolve()
    report.mkdir(parents=True, exist_ok=True)
    suite.REPORT = report
    results = {}
    for name, run_name in (('born-again', 'g2-student-s37'),
                           ('direct', 'direct-student-s37')):
        source = source_checkpoint(run_root / run_name)
        calibration = json.loads((report / name / 'calibration.json').read_text())
        if calibration['source_sha256'] != sha(source['checkpoint']):
            raise RuntimeError(f'Changed calibration source: {name}')
        settings = {key: calibration['best'][key]
                    for key in ('neural_temperature', 'cache_theta', 'cache_weight')}
        candidate = report / f'{name}-fast' / 'checkpoint.pt'
        export(source['checkpoint'], candidate, settings)
        measured = suite.bracket(f'{name}-fast', candidate)
        previous = json.loads((report / name / 'result.json').read_text())
        bpb_delta = measured['validation']['bpb'] - previous['validation']['bpb']
        if abs(bpb_delta) > 1e-5:
            raise RuntimeError(f'Inference variant changed validation BPB: {name}: {bpb_delta}')
        size = asset_bytes(candidate)
        results[name] = dict(checkpoint=str(candidate.relative_to(ROOT)),
                             checkpoint_sha256=sha(candidate),
                             implementation_sha256=sha(ROOT / f'{IMPLEMENTATION}.py'),
                             cpu_validation_bpb=measured['validation']['bpb'],
                             bpb_delta_from_training_implementation=bpb_delta,
                             baseline_before_seconds=measured['baseline_before_seconds'],
                             baseline_after_seconds=measured['baseline_after_seconds'],
                             candidate_seconds=measured['validation']['seconds'],
                             cpu_ratio_fastest=measured['cpu_ratio_fastest'],
                             cpu_ratio_mean=measured['cpu_ratio_mean'],
                             peak_ram_gib=measured['resource']['peak_working_set_gib'],
                             inference_asset_bytes=size,
                             resource_pass=measured['resource_validation_proxy_pass']
                             and size <= 64 * 1024 ** 2)
    feasible = [(name, row) for name, row in results.items() if row['resource_pass']]
    selected = min(feasible, key=lambda item: item[1]['cpu_validation_bpb'])[0] if feasible else None
    summary = dict(results=results, selected=selected,
                   old_best_validation_bpb=1.468614158,
                   selection_split='validation', test_evaluated=False)
    (report / 'optimized-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if selected:
        selection = dict(method=selected, **results[selected],
                         selection_split='validation', test_evaluated=False)
        target = report / 'frozen-selection.json'
        if target.exists() and json.loads(target.read_text()) != selection:
            raise RuntimeError('Frozen validation selection changed')
        target.write_text(json.dumps(selection, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

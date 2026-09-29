"""Select, calibrate, and resource-check the born-again and direct students.

Selection uses validation only. The fixed course evaluator scores the exported
single-student checkpoints on CPU FP32, bracketed by baseline runs.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from common import load_data, make_model, setup, sha, windows  # noqa: E402
from evaluate import score  # noqa: E402
from student import causal_cache_probs  # noqa: E402
import resource_bracket as suite  # noqa: E402


def source_checkpoint(run):
    metrics = json.loads((run / 'metrics.json').read_text())
    rows = [(row['step'], run / f"checkpoint-step{row['step']}.pt",
             min(row['neural']['bpb'], row['cached']['bpb']))
            for row in metrics['validation_history']]
    rows.append(('average', run / 'checkpoint-average.pt',
                 min(metrics['average_validation']['neural']['bpb'],
                     metrics['average_validation']['cached']['bpb'])))
    step, path, bpb = min(rows, key=lambda item: item[2])
    return dict(step=step, checkpoint=path, selection_bpb=bpb)


def export(source, destination, settings):
    saved = torch.load(source, map_location='cpu', weights_only=True)
    config = dict(saved['config'], **settings)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(saved, config=config, inference_export=dict(
            source=str(source.relative_to(ROOT)), source_sha256=sha(source),
            additional_train_targets=0, test_evaluated=False)), destination)
    exported = torch.load(destination, map_location='cpu', weights_only=True)
    if (exported['config'] != config
            or exported['inference_export']['source_sha256'] != sha(source)
            or any(not torch.equal(tensor, exported['model'][key])
                   for key, tensor in saved['model'].items())):
        raise RuntimeError(f'Changed inference export: {destination}')
    return destination


@torch.no_grad()
def calibrate(source, destination):
    device, _ = setup('cuda', 'fp32', 4)
    saved = torch.load(source, map_location='cpu', weights_only=True)
    model, _ = make_model(saved['implementation'], saved['config'], device)
    model.load_state_dict(saved['model'])
    model.eval()
    data = load_data()
    points = [(temperature, theta, weight)
              for temperature in (1., 1.05, 1.1)
              for theta in (10., 13., 16.)
              for weight in (.05, .085, .12)] + [(1., 13., 0.)]
    sums = [0.] * len(points)
    targets_count = 0
    for inputs, targets in windows(data['validation'][0]):
        inputs, targets = inputs.to(device), targets.to(device)
        features = model.features(inputs)
        logits = model.head(features).float()
        indices = targets.clamp_min(0).unsqueeze(-1)
        neural = {temperature: F.log_softmax(logits / temperature, -1).gather(
            -1, indices).squeeze(-1) for temperature in (1., 1.05, 1.1)}
        caches = {theta: causal_cache_probs(features, inputs, theta).gather(
            -1, indices).squeeze(-1).clamp_min(1e-38).log()
                  for theta in (10., 13., 16.)}
        for index, (temperature, theta, weight) in enumerate(points):
            logp = neural[temperature]
            if weight:
                logp = torch.logaddexp(logp + math.log1p(-weight),
                                       caches[theta] + math.log(weight))
                logp[:, 0] = neural[temperature][:, 0]
            sums[index] += float((-logp).masked_fill(targets == -100, 0).double().sum())
        targets_count += int((targets != -100).sum())
    if targets_count != 376599:
        raise RuntimeError('Incomplete validation calibration')
    rows = [dict(neural_temperature=temperature, cache_theta=theta, cache_weight=weight,
                 bpb=nll / math.log(2) / 1148007)
            for (temperature, theta, weight), nll in zip(points, sums)]
    rows.sort(key=lambda row: row['bpb'])
    best = rows[0]
    settings = {key: best[key] for key in ('neural_temperature', 'cache_theta', 'cache_weight')}
    checkpoint = export(source, destination, settings)
    predictor, _ = make_model(saved['implementation'], dict(saved['config'], **settings), device)
    predictor.load_state_dict(saved['model'])
    actual = score(predictor, *data['validation'], device, 'fp32')['bpb']
    if abs(actual - best['bpb']) > 2e-7:
        raise RuntimeError('Calibration differs from course scorer')
    return dict(source=str(source.relative_to(ROOT)), source_sha256=sha(source),
                points=rows, best=best, verified_gpu_bpb=actual,
                checkpoint=str(checkpoint.relative_to(ROOT)), test_evaluated=False)


def inference_asset_bytes(checkpoint):
    support = ('student_qknorm_rotary.py', 'common.py', 'evaluate.py',
               'requirements.txt', 'data/tokenizer.json', 'data/manifest.json')
    return checkpoint.stat().st_size + sum((ROOT / name).stat().st_size for name in support)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    report = ROOT / 'reports' / run_root.name
    report.mkdir(parents=True, exist_ok=True)
    suite.REPORT = report
    results = {}
    for name, run_name in (('born-again', 'g2-student-s37'),
                           ('direct', 'direct-student-s37')):
        source = source_checkpoint(run_root / run_name)
        destination = report / name / 'checkpoint.pt'
        calibration_path = report / name / 'calibration.json'
        if calibration_path.exists():
            calibration = json.loads(calibration_path.read_text())
            if calibration['source_sha256'] != sha(source['checkpoint']):
                raise RuntimeError(f'Changed calibration source: {name}')
            settings = {key: calibration['best'][key]
                        for key in ('neural_temperature', 'cache_theta', 'cache_weight')}
            export(source['checkpoint'], destination, settings)
        else:
            calibration = calibrate(source['checkpoint'], destination)
            calibration_path.parent.mkdir(parents=True, exist_ok=True)
            calibration_path.write_text(json.dumps(calibration, indent=2) + '\n')
        measured = suite.bracket(name, destination)
        asset_bytes = inference_asset_bytes(destination)
        results[name] = dict(source_step=source['step'], source_validation_bpb=source['selection_bpb'],
                             calibration=calibration['best'], cpu_validation_bpb=measured['validation']['bpb'],
                             cpu_ratio_fastest=measured['cpu_ratio_fastest'],
                             peak_ram_gib=measured['resource']['peak_working_set_gib'],
                             inference_asset_bytes=asset_bytes,
                             resource_pass=measured['resource_validation_proxy_pass']
                             and asset_bytes <= 64 * 1024 ** 2,
                             checkpoint=str(destination.relative_to(ROOT)))
    feasible = [(name, row) for name, row in results.items() if row['resource_pass']]
    selected = min(feasible, key=lambda item: item[1]['cpu_validation_bpb'])[0] if feasible else None
    summary = dict(results=results, selected=selected, selection_split='validation',
                   test_evaluated=False)
    (report / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

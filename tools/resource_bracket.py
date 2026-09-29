"""CPU FP32 validation and resource measurements for the final model."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/born-again-qknorm348-19200'
CPU = ROOT / '.venv-cpu/Scripts/python.exe'
BASELINE = ROOT / 'runs/baseline-s17/checkpoint.pt'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measured(prefix, command):
    resource = Path(str(prefix) + '.resources.json')
    if resource.exists():
        if read(resource)['exit_code'] != 0:
            raise RuntimeError(f'Previously failed process: {prefix}')
        return
    log = Path(str(prefix) + '.log')
    if log.exists():
        raise RuntimeError(f'Partial invocation exists; inspect before restarting: {prefix}')
    full = [str(CPU), 'tools/measure_run.py', '--record-prefix', str(prefix),
            '--', *map(str, command)]
    print(json.dumps({'event': 'start', 'prefix': str(prefix), 'command': full}), flush=True)
    subprocess.run(full, cwd=ROOT, check=True)


def evaluation(prefix, checkpoint):
    output = Path(str(prefix) + '-validation.json')
    measured(prefix, ['evaluate.py', '--checkpoint', checkpoint, '--device', 'cpu',
                      '--precision', 'fp32', '--threads', '4', '--split', 'validation',
                      '--output', output])
    return read(output)


def bracket(name, checkpoint):
    folder = REPORT / name
    folder.mkdir(parents=True, exist_ok=True)
    before = evaluation(folder / 'baseline-before', BASELINE)
    candidate = evaluation(folder / 'formal-cpu', checkpoint)
    after = evaluation(folder / 'baseline-after', BASELINE)
    resource = read(folder / 'formal-cpu.resources.json')
    loss = np.load(folder / 'formal-cpu-validation.window-nll.npy')
    assert len(loss) == 1472 and np.isfinite(loss).all()
    assert candidate['targets'] == 376599 and candidate['utf8_bytes'] == 1148007
    assert candidate['device'] == 'cpu' and candidate['precision'] == 'fp32'
    assert candidate['split'] == 'validation'
    assert math.isclose(float(loss.sum()), candidate['nll_nats'], abs_tol=1e-6)
    assert math.isclose(float(loss.sum()) / math.log(2) / 1148007,
                        candidate['bpb'], abs_tol=1e-12)
    assert digest(checkpoint) == candidate['checkpoint_sha256']
    baseline_mean = (before['seconds'] + after['seconds']) / 2
    size = Path(checkpoint).stat().st_size
    row = {'name': name, 'checkpoint': str(Path(checkpoint).relative_to(ROOT)),
           'validation': candidate, 'resource': resource,
           'baseline_before_seconds': before['seconds'],
           'baseline_after_seconds': after['seconds'],
           'cpu_ratio_mean': candidate['seconds'] / baseline_mean,
           'cpu_ratio_fastest': candidate['seconds'] / min(before['seconds'], after['seconds']),
           'checkpoint_bytes': size, 'window_audit_passed': True,
           'resource_validation_proxy_pass':
               candidate['seconds'] / min(before['seconds'], after['seconds']) <= 5
               and resource['peak_working_set_gib'] <= 4
               and size <= 64 * 1024 ** 2}
    (folder / 'result.json').write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'result', 'name': name, 'bpb': candidate['bpb'],
                      'cpu_ratio_mean': row['cpu_ratio_mean'],
                      'cpu_ratio_fastest': row['cpu_ratio_fastest']}), flush=True)
    return row

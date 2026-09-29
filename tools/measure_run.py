"""Run a Python script/module and record Windows peak RAM without editing it.

Example, from DASE7506_MP1/:
  .venv/Scripts/python.exe tools/measure_run.py --record-prefix reports/train -- train.py ...
Each invocation must use a fresh process and a new record prefix.
"""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
import time
import traceback


class MemoryCounters(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
    ]


def memory_counters():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        'peak_working_set_bytes': counters.PeakWorkingSetSize,
        'peak_working_set_mib': counters.PeakWorkingSetSize / 2**20,
        'peak_working_set_gib': counters.PeakWorkingSetSize / 2**30,
        'peak_commit_bytes': counters.PeakPagefileUsage,
        'peak_commit_mib': counters.PeakPagefileUsage / 2**20,
    }


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        return self.stream.write(text)

    def flush(self):
        self.log.flush()
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-prefix', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Supply a script or -m module after --.')
    if os.name != 'nt':
        parser.error('This resource recorder uses the Windows process memory API.')
    resource_path = Path(str(args.record_prefix) + '.resources.json')
    log_path = Path(str(args.record_prefix) + '.log')
    if resource_path.exists() or log_path.exists():
        parser.error('Record prefix already exists; use a new prefix.')
    resource_path.parent.mkdir(parents=True, exist_ok=True)
    memory_counters()  # Fail before running expensive work if unavailable.
    started_at = datetime.now(timezone.utc).isoformat()
    original_stdout, original_stderr = sys.stdout, sys.stderr
    original_argv, original_path = sys.argv[:], sys.path[:]
    exit_code = 0
    with log_path.open('x', encoding='utf-8') as log:
        sys.stdout, sys.stderr = Tee(original_stdout, log), Tee(original_stderr, log)
        started, cpu_started = time.perf_counter(), time.process_time()
        try:
            if command[0] == '-m':
                sys.path.insert(0, str(Path.cwd()))
                sys.argv = command[1:]
                runpy.run_module(command[1], run_name='__main__', alter_sys=True)
            else:
                sys.path.insert(0, str(Path(command[0]).resolve().parent))
                sys.argv = command[:]
                runpy.run_path(command[0], run_name='__main__')
        except SystemExit as exc:
            exit_code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
            if isinstance(exc.code, str):
                print(exc.code, file=sys.stderr)
        except BaseException:
            exit_code = 1
            traceback.print_exc()
        finally:
            elapsed = time.perf_counter() - started
            cpu_seconds = time.process_time() - cpu_started
            memory = memory_counters()
            sys.stdout, sys.stderr = original_stdout, original_stderr
            sys.argv, sys.path[:] = original_argv, original_path
    result = {
        'started_at_utc': started_at,
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'command': [sys.executable, *command],
        'command_line': subprocess.list2cmdline([sys.executable, *command]),
        'cwd': str(Path.cwd()),
        'exit_code': exit_code,
        'wall_seconds_including_imports_and_loading': elapsed,
        'process_cpu_seconds_all_threads': cpu_seconds,
        **memory,
        'memory_method': 'Windows GetProcessMemoryInfo lifetime high-water marks; fresh process per command; includes Python, imports, loading and target execution. Peak working set is resident RAM; peak commit is separate.',
        'timing_method': 'Target wall and process CPU time include imports/loading, exclude recorder setup/report writing. Use original metrics.json/evaluation seconds for pure training/scoring.',
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'logical_cpu_count': os.cpu_count(),
        'packages': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'tokenizers')},
    }
    resource_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'resource_record': str(resource_path), 'exit_code': exit_code,
                      'wall_seconds': elapsed, **memory}), flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())

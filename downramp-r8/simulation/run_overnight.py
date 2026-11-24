"""Run R8 followed by C4 in the active FBPIC environment.

    python -u run_overnight.py
    python -u run_overnight.py --resume
    python -u run_overnight.py --parallel --threads 10
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading


def plan_runs(args):
    """Resolve independent output paths before starting either simulation."""
    plan = []
    cases = ('R8',) if args.r8_only else ('R8', 'C4')
    for index, case in enumerate(cases):
        run_dir = args.output_root.resolve() / case
        threads = args.threads
        if args.parallel and len(cases) == 2:
            threads = (args.threads+1)//2 if index == 0 else args.threads//2
        restart = False
        skip = False
        if run_dir.exists() and any(run_dir.iterdir()):
            if not args.resume:
                raise ValueError(f'{run_dir} is not empty. Use --resume, or '
                                 'select a new --output-root.')
            completed = run_dir / 'completed.json'
            if completed.is_file():
                skip = json.loads(completed.read_text()).get('iteration', 0) >= 10800
            if not skip:
                restart = any((run_dir/'checkpoints').rglob('*.h5'))
                if not restart:
                    raise ValueError(f'{run_dir} has no checkpoint to resume. '
                                     'Choose a new output root for a fresh run.')
        plan.append((case, run_dir, threads, restart, skip))
    return plan


def run_case(item, script, root, dry_run, print_lock):
    case, run_dir, threads, restart, skip = item
    if skip:
        print(f'[{case}] Already complete; skipping.', flush=True)
        return 0
    cmd = [sys.executable, '-u', str(script), '--case', case,
           '--output-dir', str(run_dir)]
    if restart:
        cmd.append('--restart')
    if dry_run:
        cmd.append('--dry-run')
    env = os.environ.copy()
    env['NUMBA_NUM_THREADS'] = str(threads)
    env['OMP_NUM_THREADS'] = str(threads)
    # Avoid additional BLAS thread pools competing with Numba/FFT threads.
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    print(f'[{case}] {threads} CPU threads; '
          f'{"resume" if restart else "fresh start"}; output: {run_dir}', flush=True)
    if dry_run:
        return subprocess.run(cmd, env=env, check=False).returncode
    logs = root / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    # Append on resume, preserving earlier timings and errors.
    with (logs/f'{case}.log').open('a', encoding='utf-8') as log:
        log.write('\n=== '+('RESUME' if restart else 'NEW RUN')+' ===\n')
        log.flush()
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   errors='replace', bufsize=1)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                with print_lock:
                    print(f'[{case}] {line}', end='', flush=True)
            return process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-root', type=Path,
                        default=Path(__file__).resolve().parent / 'downramp_comparison')
    parser.add_argument('--threads', type=int, default=10,
                        help='Total thread budget; default 10.')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--r8-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.threads < (2 if args.parallel and not args.r8_only else 1):
        parser.error('Not enough threads for the selected number of processes.')
    script = Path(__file__).resolve().with_name('lwfa_r8.py')
    if not script.is_file():
        parser.error('Keep run_overnight.py and lwfa_r8.py in the same directory.')
    try:
        plan = plan_runs(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if not args.dry_run and importlib.util.find_spec('fbpic') is None:
        parser.error('FBPIC is not available in this Python environment. '
                     'Activate the existing fbpic Conda environment first.')
    print('Each case: 10800 cycles, 1.801 ps, 540 um travel; no planned stop.',
          flush=True)
    print('Mode: '+('parallel, shared thread budget' if args.parallel else
                    'sequential: R8 followed automatically by C4'), flush=True)
    lock = threading.Lock()
    results = {}
    if args.parallel and not args.dry_run:
        def worker(item):
            try:
                results[item[0]] = run_case(
                    item, script, args.output_root.resolve(), False, lock)
            except Exception as exc:
                print(f'[{item[0]}] Launcher error: {exc}', flush=True)
                results[item[0]] = 1
        workers = [threading.Thread(target=worker, args=(item,)) for item in plan]
        for worker_thread in workers:
            worker_thread.start()
        for worker_thread in workers:
            worker_thread.join()
    else:
        for item in plan:
            results[item[0]] = run_case(
                item, script, args.output_root.resolve(), args.dry_run, lock)
            # A failed case is reported, and the independent next case can run.
    print('Exit codes: '+', '.join(f'{case}={code}' for case,code in results.items()),
          flush=True)
    return int(any(code != 0 for code in results.values()))


if __name__ == '__main__':
    raise SystemExit(main())

"""Run the two full-length R8 convergence cases sequentially.

    python run_r8_convergence.py --dry-run
    python -u run_r8_convergence.py
    python -u run_r8_convergence.py --resume

Order: Nm=2, p_nt=16; then Nm=3, p_nt=16. No C4 case is launched.
The second case starts automatically only after successful completion of
the first. A failed or interrupted case stops the batch. --resume skips
verified completed cases and restarts unfinished cases from checkpoints.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


HERE = Path(__file__).resolve().parent
CASE_SCRIPT = HERE / 'cases' / 'r8_convergence_case.py'
CASES = ((2, 'R8_Nm2_nt16'), (3, 'R8_Nm3_nt16'))


def case_module():
    spec = importlib.util.spec_from_file_location('r8_convergence_case', CASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def expected_config(module, nm):
    module.set_case(nm)
    module.validate_setup()
    return module.configuration('R8')


def check_completed(run_dir, expected):
    """A zero exit code alone is not sufficient to advance the batch."""
    if read_json(run_dir / 'run_config.json') != expected:
        raise ValueError(f'{run_dir}: saved configuration does not match this case.')
    record = read_json(run_dir / 'completed.json')
    target = expected['N_step']
    if record.get('case') != 'R8' or record.get('iteration') != target:
        raise ValueError(f'{run_dir}: no verified completion at iteration {target}.')
    actual_time = float(record.get('simulation_time_s', -1))
    if abs(actual_time - target * expected['dt']) > expected['dt'] * .01:
        raise ValueError(f'{run_dir}: completion time does not match the target.')
    return record


def plan_runs(args, module):
    plan = []
    for nm, label in CASES:
        expected = expected_config(module, nm)
        run_dir = args.output_root / label
        action = 'start'
        if run_dir.exists():
            if not run_dir.is_dir():
                raise ValueError(f'{run_dir} is not a directory.')
            if any(run_dir.iterdir()):
                if not args.resume:
                    raise ValueError(f'{run_dir} is occupied. Use --resume to continue '
                                     'this study, or choose another --output-root.')
                if read_json(run_dir / 'run_config.json') != expected:
                    raise ValueError(f'{run_dir}: saved settings differ; no files changed.')
                if (run_dir / 'completed.json').is_file():
                    check_completed(run_dir, expected)
                    action = 'skip'
                elif any((run_dir / 'checkpoints').rglob('*.h5')):
                    action = 'resume'
                else:
                    raise ValueError(f'{run_dir}: unfinished with no saved checkpoint. '
                                     'Choose another --output-root for a fresh study.')
        plan.append(dict(nm=nm, label=label, run_dir=run_dir,
                         expected=expected, action=action))
    return plan


def child_environment(threads):
    env = os.environ.copy()
    env.update(NUMBA_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONUNBUFFERED='1')
    return env


def command(item, dry_run=False):
    cmd = [sys.executable, '-u', str(CASE_SCRIPT), '--nm', str(item['nm']),
           '--output-dir', str(item['run_dir'])]
    if item['action'] == 'resume':
        cmd.append('--restart')
    if dry_run:
        cmd.append('--dry-run')
    return cmd


def stop_child(process):
    if process.poll() is not None:
        return
    for sig, seconds in ((signal.SIGINT, 15), (signal.SIGTERM, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=seconds)
            return
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_case(item, root, env):
    label = item['label']
    if item['action'] == 'skip':
        print(f'[{label}] Verified complete; skipping.', flush=True)
        return 0
    path = root / 'logs' / f'{label}.log'
    print(f'[{label}] {item["action"]}; output: {item["run_dir"]}', flush=True)
    print(f'[{label}] Log: {path}', flush=True)
    with path.open('a', encoding='utf-8') as log:
        log.write(f'\n{datetime.now(timezone.utc).isoformat()} {item["action"]}\n')
        log.flush()
        process = subprocess.Popen(
            command(item), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, errors='replace',
            bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f'[{label}] {line}', end='', flush=True)
            code = process.wait()
        except BaseException:
            stop_child(process)
            raise
    if code:
        print(f'[{label}] Stopped with exit code {code}. The next case will not start.',
              flush=True)
        return code
    check_completed(item['run_dir'], item['expected'])
    print(f'[{label}] Full run completed and checked.', flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-root', type=Path, default=HERE.parent / 'runs',
                        help='Separate study output root; default: runs beside this script.')
    parser.add_argument('--threads', type=int, default=min(10, os.cpu_count() or 1),
                        help='CPU thread budget per case; default: up to 10.')
    parser.add_argument('--resume', action='store_true',
                        help='Skip completed cases and continue unfinished checkpoints.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Check and print both cases without FBPIC or file writes.')
    args = parser.parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    if args.threads < 1:
        parser.error('--threads must be positive.')
    for variable in ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'MV2_COMM_WORLD_SIZE'):
        if int(os.environ.get(variable, '1')) > 1:
            parser.error('Launch once in one CPU process; do not use mpirun.')
    try:
        module = case_module()
        plan = plan_runs(args, module)
        env = child_environment(args.threads)
        print('R8 only: Nm=2, p_nt=16, then Nm=3, p_nt=16.', flush=True)
        print('Each case: 10800 steps; 1.801246 ps; 540 um window travel.', flush=True)
        print('Automatic sequential execution; no confirmation between cases.', flush=True)
        print(f'CPU threads per case: {args.threads}', flush=True)
        if args.dry_run:
            for item in plan:
                print(f'\n[{item["label"]}] Planned action: {item["action"]}', flush=True)
                result = subprocess.run(command(item, dry_run=True), env=env, check=False)
                if result.returncode:
                    return result.returncode
            return 0
        if all(item['action'] == 'skip' for item in plan):
            print('Both cases already completed; nothing to run.', flush=True)
            return 0
        if importlib.util.find_spec('fbpic') is None:
            parser.error('FBPIC is unavailable. Activate your existing fbpic Conda environment.')
        args.output_root.mkdir(parents=True, exist_ok=True)
        with (args.output_root / '.batch.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('Another launcher is already using this output root.')
            # Recheck under the lock before creating or starting either case.
            plan = plan_runs(args, module)
            (args.output_root / 'logs').mkdir(exist_ok=True)
            (args.output_root / 'study_config.json').write_text(json.dumps({
                'case_order': [item['label'] for item in plan],
                'reference': read_json(module.reference_path()),
                'cases': {item['label']: item['expected'] for item in plan},
                'threads_per_case': args.threads,
            }, indent=2) + '\n')
            for item in plan:
                code = run_case(item, args.output_root, env)
                if code:
                    return 1
        print('Both full-length R8 convergence cases completed.', flush=True)
        return 0
    except KeyboardInterrupt:
        print('\nBatch interrupted. Use --resume to continue from saved checkpoints.',
              flush=True)
        return 130
    except (OSError, ValueError, ImportError) as exc:
        print(f'Cannot continue: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

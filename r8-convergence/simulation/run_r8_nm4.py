"""Start or resume the full Nm=4, p_nt=16 R8 run using up to ten CPU threads.

    python -u run_r8_nm4.py
    python run_r8_nm4.py --dry-run

Repeated calls resume a matching unfinished run from its latest checkpoint,
or skip a verified completed run. Existing Nm=2/3 runs are comparison inputs.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CASE_SCRIPT = HERE/'cases'/'r8_nm4_case.py'
LABEL = 'R8_Nm4_nt16'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def case_module():
    spec = importlib.util.spec_from_file_location('r8_nm4_case', CASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.set_case(4)
    module.validate_setup()
    return module


def check_completed(run_dir, expected):
    if read_json(run_dir/'run_config.json') != expected:
        raise ValueError(f'{run_dir}: configuration differs from the expected input.')
    record = read_json(run_dir/'completed.json')
    actual_time = float(record.get('simulation_time_s', -1))
    if (record.get('case') != 'R8' or record.get('iteration') != expected['N_step']
            or not math.isfinite(actual_time)
            or abs(actual_time-expected['N_step']*expected['dt']) > .01*expected['dt']):
        raise ValueError(f'{run_dir}: no verified full-length completion record.')
    return record


def plan(output_root, module):
    expected = module.configuration('R8')
    reference = dict(expected, Nm=3)
    reference_dir = output_root/'R8_Nm3_nt16'
    check_completed(reference_dir, reference)
    run_dir = output_root/LABEL
    action = 'start'
    if run_dir.exists():
        if not run_dir.is_dir():
            raise ValueError(f'{run_dir} is not a directory.')
        if any(run_dir.iterdir()):
            if read_json(run_dir/'run_config.json') != expected:
                raise ValueError(f'{run_dir}: occupied by different settings; no simulation files changed.')
            if module.prepared(run_dir):
                action = 'start'
            elif (run_dir/'completed.json').is_file():
                check_completed(run_dir, expected)
                action = 'skip'
            elif any((run_dir/'checkpoints').rglob('*.h5')):
                action = 'resume'
            else:
                raise ValueError(f'{run_dir}: unfinished with no saved checkpoint. '
                                 'Retain this folder and choose another --output-root containing '
                                 'the completed Nm=3 reference for a fresh start.')
    return dict(expected=expected, reference=reference, reference_dir=reference_dir,
                run_dir=run_dir, action=action)


def child_environment(threads):
    env = os.environ.copy()
    env.update(NUMBA_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    return env


def command(item, dry_run=False):
    cmd = [sys.executable, '-u', str(CASE_SCRIPT), '--output-dir', str(item['run_dir'])]
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
        return
    process.wait()


def run_case(item, output_root, env):
    log_path = output_root/'logs'/f'{LABEL}.log'
    print(f'Action: {item["action"]}; output: {item["run_dir"]}', flush=True)
    print(f'Log: {log_path}', flush=True)
    with log_path.open('a', encoding='utf-8') as log:
        log.write(f'\n{datetime.now(timezone.utc).isoformat()} {item["action"]}\n')
        log.flush()
        process = subprocess.Popen(command(item), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1,
            start_new_session=True)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            code = process.wait()
        except BaseException:
            stop_child(process)
            raise
    if code:
        print(f'Run stopped with exit code {code}. Rerun this launcher to use a saved checkpoint.', flush=True)
        return code
    check_completed(item['run_dir'], item['expected'])
    print('Nm=4 R8 run completed and checked at iteration 10800.', flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-root', type=Path, default=HERE.parent/'runs')
    parser.add_argument('--threads', type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument('--dry-run', action='store_true', help='Validate without FBPIC or file writes.')
    args = parser.parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    if not 1 <= args.threads <= 10:
        parser.error('--threads must be between 1 and 10.')
    for name in ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'MV2_COMM_WORLD_SIZE'):
        if int(os.environ.get(name, '1')) > 1:
            parser.error('Use one CPU process; do not launch with mpirun.')
    try:
        module = case_module()
        item = plan(args.output_root, module)
        env = child_environment(args.threads)
        print('R8: Nm=4, p_nt=16. Relative to Nm=3, only Nm changes.', flush=True)
        print('10800 steps; 1.801246 ps; 540 um window travel.', flush=True)
        print(f'CPU threads: {args.threads}; planned action: {item["action"]}.', flush=True)
        if args.dry_run:
            return subprocess.call(command(item, dry_run=True), env=env)
        if item['action'] == 'skip':
            print('Verified complete; nothing to run.', flush=True)
            return 0
        if importlib.util.find_spec('fbpic') is None:
            parser.error('FBPIC is unavailable. Activate the existing fbpic Conda environment.')
        with (args.output_root/'.batch.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('Another convergence launcher is using this output root.')
            item = plan(args.output_root, module)
            if item['action'] == 'skip':
                print('Verified complete; nothing to run.', flush=True)
                return 0
            (args.output_root/'logs').mkdir(exist_ok=True)
            # Keep the previous two-case study_config.json untouched.
            (args.output_root/'nm4_study_config.json').write_text(json.dumps(dict(
                label=LABEL, reference_run=str(item['reference_dir']),
                reference=item['reference'], configuration=item['expected'],
                changes_from_reference={'Nm': 4}, threads=args.threads), indent=2)+'\n')
            return run_case(item, args.output_root, env)
    except KeyboardInterrupt:
        print('\nRun interrupted. Rerun the same command to resume from the latest saved checkpoint.', flush=True)
        return 130
    except (OSError, ValueError, ImportError) as exc:
        print(f'Cannot run Nm=4: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

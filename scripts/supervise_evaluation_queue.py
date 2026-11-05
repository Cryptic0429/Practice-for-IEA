"""Finish the two pending frozen test conditions when an allowed GPU is idle.

Only launches Official and BaseSchema; never restarts workers, changes settings,
or stops a process. Existing Base/A/B runs must already have launch records.
"""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/mnt/sdc/zhc/IEA')
MODELS = ['Base', 'A', 'B', 'Official', 'BaseSchema']
PENDING = ['Official', 'BaseSchema']
FOLDER = ROOT / 'logs/evaluation-v1'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def write_status(data):
    path = FOLDER / 'queue-status.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def event(name, **fields):
    with (FOLDER / 'queue-events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(time=stamp(), event=name, **fields)) + '\n')


def alive(pid):
    folder = Path(f'/proc/{pid}')
    if not folder.exists():
        return False
    try:
        state = (folder / 'stat').read_text().split(') ', 1)[1].split()[0]
        command = (folder / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        return state not in ('Z', 'X') and 'scripts/evaluate_stage1.py' in command
    except FileNotFoundError:
        return False


def gpu_idle(gpu):
    occupants = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                                         '--format=csv,noheader,nounits'], text=True).strip()
    free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                                        '--format=csv,noheader,nounits'], text=True).strip())
    return not occupants and free >= 14000


def main():
    FOLDER.mkdir(exist_ok=True)
    lock = (FOLDER / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert read(ROOT / 'configs/evaluation-models-v1.json')['all_selected_models_verified']
    for model in MODELS[:3]:
        assert (ROOT / f'outputs/evaluation-v1/test-{model}-full/launch.json').is_file()
    state = dict(status='running', pid=os.getpid(), started_at=stamp(),
                 allowed_gpu_indices=[0, 2, 3], protected_gpu_indices=[1], jobs={})
    event('queue_started', pid=os.getpid())
    reported_completed = set()
    try:
        while True:
            jobs = {}
            for model in MODELS:
                folder = ROOT / f'outputs/evaluation-v1/test-{model}-full'
                launch = folder / 'launch.json'
                if not launch.exists():
                    assert model in PENDING and not (folder / 'worker.log').exists(), model
                    jobs[model] = dict(status='pending')
                    continue
                launched = read(launch)
                live = alive(launched['pid'])
                status = read(folder / 'status.json') if (folder / 'status.json').exists() else {'status':'starting'}
                jobs[model] = dict(status=status['status'], pid=launched['pid'],
                                   gpu=launched['gpu'], alive=live, completed=status.get('completed', 0))
                assert status['status'] != 'failed', f'{model} failed; inspect its preserved log.'
                if status['status'] == 'completed':
                    assert status['completed'] == status['total'] == 200, model
                    if model not in reported_completed:
                        event('condition_completed', model=model)
                        reported_completed.add(model)
                else:
                    assert live, f'{model} worker exited without complete status; do not restart automatically.'
            state.update(updated_at=stamp(), jobs=jobs)
            if all(job['status'] == 'completed' for job in jobs.values()):
                state.update(status='completed', finished_at=stamp())
                write_status(state)
                event('all_five_conditions_completed')
                return
            write_status(state)
            for model in PENDING:
                if jobs[model]['status'] != 'pending':
                    continue
                for gpu in [0, 2, 3]:
                    # Keep a launch/loading worker's card reserved before CUDA initialization.
                    if any(j.get('alive') and str(j.get('gpu')) == str(gpu) for j in jobs.values()):
                        continue
                    if not gpu_idle(gpu):
                        continue
                    command = (f'export IEA_GPU={gpu}; source {ROOT}/activate.sh; '
                               f'python {ROOT}/scripts/launch_evaluation.py '
                               f'--model {model} --split test --gpu {gpu} --run-name full')
                    result = subprocess.run(['bash', '-c', command], check=True, capture_output=True, text=True)
                    launched = json.loads(result.stdout)
                    event('condition_launched', model=model, **launched)
                    jobs[model] = dict(status='starting', alive=True, gpu=str(gpu), pid=launched['pid'])
                    break
            time.sleep(10)
    except BaseException as exc:
        state.update(status='attention_required', updated_at=stamp(), error=f'{type(exc).__name__}: {exc}')
        write_status(state)
        event('queue_stopped_for_inspection', error=state['error'])
        raise


if __name__ == '__main__':
    main()

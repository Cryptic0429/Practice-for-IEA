"""Run the frozen experiment once and keep durable, atomic monitoring records."""
import csv
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path('/mnt/sdc/zhc/IEA')
RUN = ROOT / 'logs/stage1-A-3k'
OUT = ROOT / 'outputs/stage1-A-image-edit-3k-v1'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def event(kind, **details):
    with (RUN / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(time=stamp(), event=kind, **details), ensure_ascii=False) + '\n')


def records():
    path = OUT / 'trainer_log.jsonl'
    result = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # The writer may still be appending the final line.
    return result


def finite_json(obj):
    if isinstance(obj, dict):
        return {k: finite_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [finite_json(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    return obj


def verify_artifacts():
    from safetensors.torch import load_file
    import torch

    required = ['adapter_config.json', 'adapter_model.safetensors',
                'trainer_state.json', 'train_results.json', 'eval_results.json']
    for name in required:
        assert (OUT / name).is_file(), name
    state = json.loads((OUT / 'trainer_state.json').read_text())
    assert state['global_step'] == 376, state['global_step']
    best = Path(state['best_model_checkpoint']).resolve()
    assert best.parent == OUT.resolve(), best
    assert math.isfinite(state['best_metric']), state['best_metric']
    final = load_file(str(OUT / 'adapter_model.safetensors'))
    selected = load_file(str(best / 'adapter_model.safetensors'))
    assert final.keys() == selected.keys()
    for name in final:
        assert torch.isfinite(final[name]).all(), name
        assert torch.equal(final[name], selected[name]), name
    report = dict(verified_at=stamp(), global_step=state['global_step'],
                  epoch=state.get('epoch'), best_checkpoint=str(best),
                  best_validation_loss=state['best_metric'],
                  final_equals_best_checkpoint=True, adapter_tensor_count=len(final),
                  adapter_sha256=hashlib.sha256((OUT / 'adapter_model.safetensors').read_bytes()).hexdigest(),
                  test_set_evaluated=False, full_model_reload_verified=False)
    write_json(RUN / 'artifact-verification.json', report)
    return report


def main():
    global RUN, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=['A', 'B'], default='A')
    parser.add_argument('--attach', action='store_true', help='Monitor the existing trainer; never launch another.')
    args = parser.parse_args()
    if args.experiment == 'B':
        RUN = ROOT / 'logs/stage1-B-3k'
        OUT = ROOT / 'outputs/stage1-B-image-edit-3k-v1'
    config_path = ROOT / f'configs/qlora-stage1-{args.experiment}-3k.yaml'
    launch_path = ROOT / ('scripts/train_stage1_B_frozen.sh' if args.experiment == 'B' else 'scripts/train_stage1_frozen.sh')
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.attach and ((RUN / 'run.json').exists() or OUT.exists()):
        raise RuntimeError('Run already exists; inspect it before any restart.')
    env = dict(os.environ, IEA_GPU='2', PYTHONUNBUFFERED='1', RECORD_VRAM='1')
    started = time.time()
    if args.attach:
        run = json.loads((RUN / 'run.json').read_text())
        pid = run['trainer_pid']
        proc = Path(f'/proc/{pid}')
        command = (proc / 'cmdline').read_bytes()
        assert b'llamafactory-cli' in command and str(config_path).encode() in command
        start_ticks = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]

        class Attached:
            def __init__(self):
                self.pid = pid

            def poll(self):
                try:
                    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
                    return None if fields[19] == start_ticks and fields[0] != 'Z' else -256
                except FileNotFoundError:
                    return -256  # Exit status cannot be reaped by a new parent.

        child = Attached()
        run.pop('gpu', None)
        run.update(supervisor_pid=os.getpid(), gpu_index=2, supervisor_reattached_at=stamp())
        started = datetime.fromisoformat(run['started_at']).timestamp()
        event('supervisor_reattached', trainer_pid=pid, supervisor_pid=os.getpid())
    else:
        with (RUN / 'train.log').open('xb', buffering=0) as log:
            child = subprocess.Popen(['bash', str(launch_path)],
                                     env=env, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        run = dict(started_at=stamp(), supervisor_pid=os.getpid(), trainer_pid=child.pid,
                   gpu_index=2, output_dir=str(OUT), log_dir=str(RUN), poll_seconds=10,
                   config=str(config_path), experiment=args.experiment,
                   configuration_changed=False, training_started=True)
        event('started', **run)
    write_json(RUN / 'run.json', run)
    seen_evals = set()
    seen_checkpoints = set()
    previous_alerts = set()
    peak_mib = 0
    stopped_for_nonfinite = False
    stop_deadline = None
    with (RUN / 'gpu.csv').open('a' if args.attach else 'x', newline='') as gpu_stream:
        writer = csv.writer(gpu_stream)
        if gpu_stream.tell() == 0:
            writer.writerow(['utc', 'memory_used_mib', 'memory_total_mib', 'utilization_percent', 'temperature_c'])
        while True:
            code = child.poll()
            items = records()
            latest = items[-1] if items else {}
            alerts = set()
            invalid_loss = any(isinstance(row.get(key), (float, int)) and not math.isfinite(row[key])
                               for row in items for key in ('loss', 'eval_loss'))
            with (RUN / 'train.log').open('rb') as stream:
                stream.seek(max(0, (RUN / 'train.log').stat().st_size - 65536))
                tail = stream.read().decode(errors='replace')
            invalid_grad = bool(re.search(r"['\"]grad_norm['\"]\s*:\s*(?:nan|[-+]?inf)\b", tail, re.I))
            if invalid_loss or invalid_grad:
                alerts.add('nonfinite_loss_or_gradient')
                if code is None and not stopped_for_nonfinite:
                    os.killpg(child.pid, signal.SIGTERM)
                    stopped_for_nonfinite = True
                    stop_deadline = time.time() + 30
                    event('terminated_owned_training_for_nonfinite_values')
            if stop_deadline is not None and code is None and time.time() > stop_deadline:
                os.killpg(child.pid, signal.SIGKILL)
                stop_deadline = None
            if code is None and time.time() - (RUN / 'train.log').stat().st_mtime > 600:
                alerts.add('no_log_activity_for_10_minutes')
            gpu = {}
            try:
                raw = subprocess.check_output([
                    'nvidia-smi', '-i', '2',
                    '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
                    '--format=csv,noheader,nounits'], text=True, timeout=8)
                used, total, utilization, temperature = map(int, raw.strip().split(','))
                gpu = dict(memory_used_mib=used, memory_total_mib=total,
                           utilization_percent=utilization, temperature_c=temperature)
                peak_mib = max(peak_mib, used)
                writer.writerow([stamp(), used, total, utilization, temperature])
                gpu_stream.flush()
                if temperature >= 90:
                    alerts.add('gpu_temperature_at_least_90C')
            except Exception as exc:
                gpu = dict(query_error=str(exc))
                alerts.add('gpu_query_failed')
            for row in items:
                if 'eval_loss' in row:
                    step = row.get('current_steps')
                    if step not in seen_evals:
                        event('validation', **finite_json(row))
                        seen_evals.add(step)
            checkpoints = sorted(str(p) for p in OUT.glob('checkpoint-*') if p.is_dir())
            for checkpoint in checkpoints:
                if checkpoint not in seen_checkpoints and (Path(checkpoint) / 'trainer_state.json').exists():
                    event('checkpoint_saved', path=checkpoint)
                    seen_checkpoints.add(checkpoint)
            for alert in alerts - previous_alerts:
                event('alert', reason=alert)
            for alert in previous_alerts - alerts:
                event('alert_cleared', reason=alert)
            previous_alerts = alerts
            state = dict(**run, updated_at=stamp(), status='running' if code is None else 'verifying',
                         elapsed_seconds=round(time.time()-started, 1), exit_code=None if code == -256 else code,
                         process_alive=code is None, exit_status_available=not args.attach,
                         latest=finite_json(latest), gpu=gpu, sampled_peak_gpu_mib=peak_mib,
                         alerts=sorted(alerts), checkpoints=checkpoints,
                         validation_history=finite_json([r for r in items if 'eval_loss' in r]))
            write_json(RUN / 'status.json', state)
            if code is not None:
                if code in (0, -256):
                    try:
                        state['artifact_verification'] = verify_artifacts()
                        state['status'] = 'completed'
                    except Exception as exc:
                        state['status'] = 'completed_verification_failed' if code == 0 else 'failed'
                        state['verification_error'] = repr(exc)
                else:
                    state['status'] = 'failed'
                state['finished_at'] = stamp()
                write_json(RUN / 'status.json', state)
                event(state['status'], exit_code=state['exit_code'], verification_error=state.get('verification_error'))
                return
            time.sleep(10)


if __name__ == '__main__':
    main()

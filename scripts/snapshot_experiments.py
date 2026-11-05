"""Read-only remote monitoring; persist compact local snapshots without job control."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

REMOTE = r'''
import ast, datetime, json, math, pathlib, re, subprocess, time
root=pathlib.Path('/mnt/sdc/zhc/IEA')
now=time.time()
def process(pid, required):
    p=pathlib.Path('/proc')/str(pid)
    try:
        command=(p/'cmdline').read_bytes().replace(b'\x00', b' ').decode()
        state=(p/'stat').read_text().split(') ', 1)[1].split()[0]
        return dict(pid=pid, alive=state!='Z' and required in command, state=state, command=command)
    except FileNotFoundError:
        return dict(pid=pid, alive=False)
def read(path): return json.loads(path.read_text())
result=dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), training={}, evaluation={})
for model in ['A','B']:
    folder=root/f'logs/stage1-{model}-3k'
    if not (folder/'status.json').exists(): continue
    status=read(folder/'status.json')
    item=dict(status=status, run=read(folder/'run.json'),
              events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()],
              supervisor=process(status['supervisor_pid'],'supervise_stage1_training.py'),
              trainer=process(status['trainer_pid'],'llamafactory-cli'),
              status_age_seconds=now-(folder/'status.json').stat().st_mtime,
              log_age_seconds=now-(folder/'train.log').stat().st_mtime)
    text=(folder/'train.log').read_text(errors='replace')
    item['log_tail']='\n'.join(text.replace('\r','\n').splitlines()[-12:])
    values=[]
    for match in re.finditer(r"\{'loss':[^\n]+\}", text):
        try:
            point=ast.literal_eval(match.group())
            values.extend(float(point[k]) for k in ['loss','grad_norm'] if k in point)
        except (ValueError,SyntaxError):
            # Nonfinite values are not Python literal numbers; make the check fail.
            if re.search(r'\b(?:nan|inf)\b',match.group(),re.I): values.append(float('nan'))
    item['finite_loss_and_gradient_values']=bool(values) and all(math.isfinite(v) for v in values)
    item['checked_loss_and_gradient_count']=len(values)
    item['checkpoint_files']={}
    for checkpoint in sorted(pathlib.Path(status['output_dir']).glob('checkpoint-*')):
        item['checkpoint_files'][checkpoint.name]={name:(checkpoint/name).is_file() for name in
            ['trainer_state.json','adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt']}
    for name in ['artifact-verification.json','reload-verification.json','summary.json']:
        if (folder/name).exists(): item[name]=read(folder/name)
    result['training'][model]=item
for path in sorted((root/'outputs/evaluation-v1').glob('*/status.json')):
    s=read(path)
    item=dict(status=s, process=process(s['pid'],'evaluate_stage1.py'),
              status_age_seconds=now-path.stat().st_mtime)
    if (path.parent/'diagnostic-summary.json').exists(): item['diagnostic_summary']=read(path.parent/'diagnostic-summary.json')
    result['evaluation'][path.parent.name]=item
queue=root/'logs/evaluation-v1/queue-status.json'
if queue.exists():
    status=read(queue)
    result['evaluation_queue']=dict(status=status, process=process(status['pid'],'supervise_evaluation_queue.py'),
                                    status_age_seconds=now-queue.stat().st_mtime)
preparation=root/'logs/evaluation-v1/preparation-status.json'
if preparation.exists():
    status=read(preparation)
    result['evaluation_preparation']=dict(status=status, process=process(status['pid'],'prepare_evaluation_outputs.py'),
                                          status_age_seconds=now-preparation.stat().st_mtime)
result['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,temperature.gpu,utilization.gpu','--format=csv,noheader'],text=True).strip().splitlines()
result['compute_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True).strip().splitlines()
print(json.dumps(result,allow_nan=False))
'''


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compact', action='store_true')
    args = parser.parse_args()
    observed = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'SJTU1', 'python3 -'],
                              input=REMOTE, capture_output=True, text=True, timeout=45)
    if observed.returncode:
        raise RuntimeError('SSH observation failed; no remote job state was changed: ' + observed.stderr)
    data = json.loads(observed.stdout)
    write(ROOT / 'metadata/experiment-latest-snapshot.json', data)
    progress_path = ROOT / 'metadata/experiment-progress.json'
    progress = json.loads(progress_path.read_text())
    progress['updated_at'] = data['observed_at']
    checks = []
    for model, item in data['training'].items():
        for key, name in [('status', 'status.json'), ('run', 'run.json')]:
            write(ROOT / f'metadata/stage1-{model}-3k-run/{name}', item[key])
        for name in ['artifact-verification.json', 'reload-verification.json', 'summary.json']:
            if name in item:
                write(ROOT / f'metadata/stage1-{model}-3k-run/{name}', item[name])
        s = item['status']
        if model in progress:
            progress[model]['latest_step'] = s.get('latest', {}).get('current_steps')
            progress[model]['last_observed_at'] = data['observed_at']
        if s['status'] == 'running':
            if not item['trainer']['alive']: checks.append(f'{model}: training PID is absent or mismatched; inspect terminal artifacts.')
            if not item['supervisor']['alive']: checks.append(f'{model}: supervisor PID is absent or mismatched.')
            if item['status_age_seconds'] > 90: checks.append(f'{model}: supervision status older than 90 seconds.')
            if item['log_age_seconds'] > 600: checks.append(f'{model}: training log idle over 10 minutes.')
            if not item['finite_loss_and_gradient_values']: checks.append(f'{model}: loss/gradient finite check failed or no values found.')
        checks.extend(f'{model}: {alert}' for alert in s.get('alerts', []))
    diagnostics = progress['evaluation'].setdefault('validation_diagnostics', {})
    for name, item in data['evaluation'].items():
        s = item['status']
        if s['split'] == 'validation':
            diagnostics[s['model']] = dict(status=s['status'], completed=s['completed'], rows=s['total'],
                                           gpu=s['gpu'], pid=s['pid'], directory=s['output_dir'])
        if s['split'] == 'test':
            progress['evaluation'].setdefault('test_jobs', {})[s['model']] = dict(
                status=s['status'], completed=s['completed'], rows=s['total'],
                gpu=s['gpu'], pid=s['pid'], directory=s['output_dir'])
            if s['status'] == 'failed': checks.append(f'{name}: test worker failed; inspect preserved error/log.')
            reviewed = progress['evaluation'].get('reviewed_inference_error_counts', {}).get(s['model'], 0)
            if s.get('inference_errors', 0) > reviewed:
                checks.append(f"{name}: {s['inference_errors']} inference exception(s) recorded; inspect without changing frozen settings.")
        if s['status'] in ('running', 'loading') and not item['process']['alive']:
            checks.append(f'{name}: evaluation PID is absent or mismatched.')
        if s['status'] in ('running', 'loading') and item['status_age_seconds'] > 600:
            checks.append(f'{name}: no evaluation progress for over 10 minutes; inspect live worker.')
    if 'evaluation_queue' in data:
        item = data['evaluation_queue']
        progress['evaluation']['queue'] = item['status']
        write(ROOT / 'metadata/evaluation-v1/queue-status.json', item['status'])
        if item['status']['status'] == 'attention_required':
            checks.append('Evaluation queue needs inspection: ' + item['status'].get('error', 'unknown'))
        if item['status']['status'] == 'running' and (not item['process']['alive'] or item['status_age_seconds'] > 90):
            checks.append('Evaluation queue stopped or stale; check jobs before restarting supervision.')
    if 'evaluation_preparation' in data:
        item = data['evaluation_preparation']
        progress['report_preparation']['worker'] = item['status']
        write(ROOT / 'metadata/evaluation-v1/preparation-status.json', item['status'])
        if item['status']['status'] == 'attention_required':
            checks.append('Report preparation needs inspection: ' + item['status'].get('error', 'unknown'))
        if item['status']['status'] in ('waiting_for_all_tests', 'preparing') and not item['process']['alive']:
            checks.append('Report preparation PID is absent or mismatched.')
    progress['evaluation']['test_inference_started'] = any(name.startswith('test-') for name in data['evaluation'])
    progress['last_observation_checks'] = checks
    write(progress_path, progress)
    monitor_path = ROOT / 'metadata/stage1-A-3k-monitor-state.json'
    monitor = json.loads(monitor_path.read_text())
    monitor.update(last_checked_at=data['observed_at'], last_check_result='; '.join(checks) if checks else 'Live process, freshness and finite-value checks passed for active jobs.')
    write(monitor_path, monitor)
    compact = dict(observed_at=data['observed_at'], checks=checks, gpus=data['gpus'],
                   training={m:dict(status=v['status']['status'], step=v['status'].get('latest',{}).get('current_steps'),
                                    trainer_alive=v['trainer']['alive'], supervisor_alive=v['supervisor']['alive'],
                                    validation_history=v['status'].get('validation_history',[])) for m,v in data['training'].items()},
                   validation={m:v for m,v in diagnostics.items()})
    if args.compact:
        brief = dict(observed_at=data['observed_at'], checks=checks,
                     training={m:dict(status=v['status']['status'],
                                      step=v['status'].get('latest',{}).get('current_steps'),
                                      loss=v['status'].get('latest',{}).get('loss'),
                                      trainer_alive=v['trainer']['alive'], supervisor_alive=v['supervisor']['alive'],
                                      validation_steps=[x['current_steps'] for x in v['status'].get('validation_history',[])],
                                      finite=v['finite_loss_and_gradient_values'],
                                      gpu=v['status'].get('gpu')) for m,v in data['training'].items() if v['status']['status'] != 'completed'},
                     evaluation={name:dict(status=v['status']['status'], count=v['status']['completed'],
                                           total=v['status']['total'], alive=v['process']['alive']) for name,v in data['evaluation'].items()
                                 if name.startswith(('validation-','test-'))})
        print(json.dumps(brief, ensure_ascii=False))
    else:
        print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()

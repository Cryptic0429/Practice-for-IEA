"""Local read-only SJTU1 observation, preserving last successful state on errors."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
META=ROOT/'metadata/stage2-A-edit-grpo-v1'


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def main():
    now=datetime.now(timezone.utc).isoformat()
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','SJTU1',
             'cd /mnt/sdc/zhc/IEA && envs/iea-rl/bin/python scripts/snapshot_stage2.py']
    try:
        process=subprocess.run(command,capture_output=True,text=True,timeout=45)
    except subprocess.TimeoutExpired:
        report=dict(observed_at=now,status='observation_timeout',training_failure_proven=False)
        write(META/'observation-error.json',report);print(json.dumps(report));return
    if process.returncode:
        report=dict(observed_at=now,status='ssh_unobservable' if process.returncode==255 else 'remote_probe_error',
                    returncode=process.returncode,stderr=process.stderr[-4000:],training_failure_proven=False)
        write(META/'observation-error.json',report);print(json.dumps(report));return
    observation=json.loads(process.stdout)
    write(META/'observation.json',observation)
    pipeline=observation.get('experiment_pipeline',{})
    active=[]
    for item in observation['active_runs']:
        metrics=item.get('last_metrics',{})
        active.append(dict(run=item['run'],pid=item['pid'],status=item.get('status'),
            step=item.get('step',metrics.get('step')),completed=item.get('completed'),
            candidates=item.get('candidates_rows'),valid=item.get('valid_candidates'),
            grad_norm=metrics.get('grad_norm'),kl=metrics.get('kl'),
            last_gpu_sample=item.get('last_gpu_sample'),sampled_peak_mib=item.get('sampled_device_peak_mib')))
    previous=read_previous()
    events=[]
    phase=pipeline.get('phase')
    if previous and previous.get('phase')!=phase:
        events.append('pipeline_phase_changed')
    if previous:
        old_runs={r['run']:r for r in previous.get('active',[])}
        new_runs={r['run']:r for r in active}
        for name in new_runs.keys()-old_runs.keys():
            if name.startswith(('validation-','test_')):
                events.append('evaluation_started:'+name)
        for name in old_runs.keys()-new_runs.keys():
            if name.startswith(('validation-','test_')):
                events.append('evaluation_process_ended_inspect_result:'+name)
        old_step=old_runs.get('formal-v1',{}).get('step') or 0
        new_step=new_runs.get('formal-v1',{}).get('step') or 0
        for boundary in [64,128]:
            if old_step<boundary<=new_step:
                events.append('formal_checkpoint_boundary_reached:'+str(boundary))
    terminal=pipeline.get('status') not in ['running','waiting_for_gpu','starting']
    if terminal or (pipeline and not pipeline.get('process_alive')):
        events.append('pipeline_terminal_or_missing_inspect')
    if observation['unknown_intervals']:
        events.append('resource_accounting_unknown')
    report=dict(observed_at=observation['observed_at'],status='observed',phase=phase,
        pipeline_status=pipeline.get('status'),supervisor_alive=pipeline.get('process_alive'),
        active=active,remaining_gpu_hours=observation['remaining_gpu_hours'],events=events)
    write(META/'poll-state.json',report)
    progress=json.loads((META/'progress.json').read_text())
    progress.update(updated_at=report['observed_at'],gpu_hours_used=observation['total_gpu_hours'])
    formal=next((item for item in active if item['run']=='formal-v1'),None)
    if formal: progress['formal_optimizer_steps']=formal['step'] or 0
    progress['last_goal_turn_classification']='verified_wait: actual process liveness and fresh finite metric/GPU evidence' if active and not events else 'observation: inspect pipeline phase or terminal evidence before choosing next action'
    write(META/'progress.json',progress)
    print(json.dumps(report,ensure_ascii=False,indent=2))


def read_previous():
    path=META/'poll-state.json'
    return json.loads(path.read_text()) if path.exists() else None


if __name__=='__main__':
    main()

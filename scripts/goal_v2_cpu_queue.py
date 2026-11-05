"""Complete predeclared CPU diagnostics as validation jobs finish; no test access."""
import datetime, json, os, subprocess, time, fcntl
from pathlib import Path
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    lock=(OUT/'cpu-queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    names=['A','B','SFT','GRPO','full-lr1e-5-s20261103','parameters-lr1e-5-s20261103','parameters-lr3e-5-s20261103']
    start=time.monotonic();status={'pid':os.getpid(),'started_at':stamp(),'status':'waiting_for_validation','completed':[]}
    def save():
        status['updated_at']=stamp();p=OUT/'cpu-queue.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(status,indent=2)+'\n');t.replace(p)
    save()
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(ROOT/'scripts'))
    while len(status['completed'])<len(names):
        progress=False
        for name in names:
            if name in status['completed']:continue
            source=OUT/'evaluation'/f'validation-{name}'
            state=source/'status.json'
            if not state.exists() or json.loads(state.read_text())['status']!='completed':continue
            destination=OUT/'calibration'/f'{name}-validation-200'
            assert not destination.exists(), 'Never overwrite a diagnostic; inspect interrupted work'
            with (OUT/f'calibration-{name}-200.log').open('x') as log:
                child=subprocess.Popen([str(ROOT/'envs/iea-rl/bin/python'),str(ROOT/'scripts/goal_v2_calibrate.py'),
                    '--predictions',str(source/'predictions.jsonl'),'--name',f'{name}-validation-200'],stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
                status.update(status='calibrating',task=name,child_pid=child.pid);save()
                while child.poll() is None:time.sleep(5);save()
                if child.returncode:status.update(status='failed',exit_code=child.returncode);save();return
            result=json.loads((destination/'summary.json').read_text());assert result['n']==200
            status['completed'].append(name);save();progress=True
        if len(status['completed'])==len(names):break
        workers=[json.loads(p.read_text()) for p in OUT.glob('worker-*.json')]
        live=[w for w in workers if Path(f"/proc/{w['pid']}").exists() and w['status']=='running']
        if not live:
            status.update(status='incomplete_workers_terminal',missing=[n for n in names if n not in status['completed']]);save();return
        if time.monotonic()-start>14400:status['status']='observation_time_limit';save();return
        status['status']='waiting_for_validation';save();time.sleep(30)
    status['status']='completed';save()
if __name__=='__main__':main()

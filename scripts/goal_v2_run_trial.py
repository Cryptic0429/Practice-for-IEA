"""One isolated development trial or seeded replication, after a verified GPU handoff."""
import argparse,datetime,fcntl,json,os,re,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',choices=['0','2','3'],required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--lr',type=float,required=True);ap.add_argument('--seed',type=int,default=20261103)
    ap.add_argument('--wait-for',required=True);ap.add_argument('--json-only',action='store_true')
    ap.add_argument('--training-file',choices=['train.json','train-contrast-calibrated.json'],default='train-contrast-calibrated.json');a=ap.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]+',a.name) and 0<a.lr<=1e-3
    assert Path(a.wait_for).name==a.wait_for and a.wait_for.startswith('worker-') and a.wait_for.endswith('.json')
    lock=(OUT/f'worker-trial-{a.name}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=OUT/f'worker-trial-{a.name}.json';assert not path.exists() and not (OUT/a.name).exists()
    status={'pid':os.getpid(),'gpu':a.gpu,'status':'waiting','started_at':stamp(),'settings':vars(a),'completed':[]};start=time.monotonic()
    def save():
        status['updated_at']=stamp();t=path.with_suffix('.tmp');t.write_text(json.dumps(status,indent=2)+'\n');t.replace(path)
    save()
    while True:
        previous=json.loads((OUT/a.wait_for).read_text())
        if previous['status']=='failed':status['status']='incomplete_previous_queue_failed';save();return
        if previous['status']=='completed':
            memory=int(subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
            if memory<512:break
        if time.monotonic()-start>14400:status['status']='observation_time_limit';save();return
        time.sleep(10)
    suffix='_json' if a.json_only else '';py=str(ROOT/'envs/iea-rl/bin/python')
    commands=[('train',[py,str(ROOT/f'scripts/goal_v2_train{suffix}.py'),'--name',a.name,'--mode','parameters','--lr',str(a.lr),'--seed',str(a.seed),'--gpu',a.gpu,'--training-file',a.training_file]),
              ('validate',[py,str(ROOT/f'scripts/goal_v2_evaluate{suffix}.py'),'--adapter',str(OUT/a.name/'final'),'--name',a.name,'--split','validation','--gpu',a.gpu,'--max-gpu-seconds','5400'])]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,PYTHONPATH=str(ROOT/'scripts'),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',XDG_CACHE_HOME=str(ROOT/'cache'),TMPDIR=str(ROOT/'tmp'))
    for task,command in commands:
        with (OUT/f'{a.name}-{task}.log').open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
            status.update(status='running',task=f'{a.name}-{task}',child_pid=child.pid,task_started_at=stamp());save()
            while child.poll() is None:time.sleep(5);save()
            status['completed'].append({'task':task,'exit_code':child.returncode,'finished_at':stamp()});save()
            if child.returncode:status['status']='failed';save();return
    state=json.loads((OUT/'evaluation'/f'validation-{a.name}'/'status.json').read_text())
    if state['status']!='completed':status['status']='incomplete_validation';save();return
    with (OUT/f'{a.name}-analysis.log').open('x') as log:
        child=subprocess.Popen([py,str(ROOT/'scripts/goal_v2_analyze_validation.py')],stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
        status.update(task=f'{a.name}-analysis',child_pid=child.pid);save()
        while child.poll() is None:time.sleep(5);save()
    status['status']='completed' if child.returncode==0 else 'failed';save()
if __name__=='__main__':main()

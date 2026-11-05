"""Small fixed-64 controls: longer q/v optimization or wider language adapter scope."""
import argparse,datetime,fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['long','expanded'],required=True);a=ap.parse_args()
    gpu='2' if a.mode=='long' else '3';path=OUT/f'worker-capacity-{a.mode}.json'
    lock=path.with_suffix('.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not path.exists()
    status={'pid':os.getpid(),'gpu':gpu,'status':'waiting','started_at':stamp(),'completed':[]};begin=time.monotonic()
    def save():
        status['updated_at']=stamp();t=path.with_suffix('.tmp');t.write_text(json.dumps(status,indent=2)+'\n');t.replace(path)
    save()
    while True:
        previous=json.loads((OUT/f'worker-memorize-{gpu}.json').read_text());ready=previous['status']=='completed'
        if a.mode=='expanded':
            verification=ROOT/'models/goal-v2-A-expanded-start-v1/start-verification.json';ready=ready and verification.exists()
            launch=OUT/'expanded-start-launch.json'
            if launch.exists() and not verification.exists():
                pid=json.loads(launch.read_text())['pid']
                if not Path(f'/proc/{pid}').exists():status['status']='initialization_preflight_failed';save();return
        if ready:
            used=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
            if used<512:break
        if time.monotonic()-begin>14400:status['status']='observation_time_limit';save();return
        time.sleep(10)
    py=str(ROOT/'envs/iea-rl/bin/python')
    if a.mode=='long':
        name='json-memorize64-long512-lr1e-4-s20261103';train='goal_v2_train_memorize64_long.py';probe='goal_v2_evaluate_memorize64.py';evaluate='goal_v2_evaluate_json.py'
    else:
        name='expanded-memorize64-lr1e-4-s20261103';train='goal_v2_train_memorize64_expanded.py';probe='goal_v2_evaluate_memorize64_expanded.py';evaluate='goal_v2_evaluate_expanded.py'
    adapter=OUT/name/'final'
    commands=[('train',[py,str(ROOT/'scripts'/train),'--name',name,'--mode','parameters','--lr','1e-4','--gpu',gpu,'--training-file','train-memorize64.json']),
              ('training-probe',[py,str(ROOT/'scripts'/probe),'--adapter',str(adapter),'--name',name,'--split','train_probe','--gpu',gpu,'--max-gpu-seconds','3600']),
              ('validate',[py,str(ROOT/'scripts'/evaluate),'--adapter',str(adapter),'--name',name,'--split','validation','--gpu',gpu,'--max-gpu-seconds','5400'])]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH=str(ROOT/'scripts'),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',XDG_CACHE_HOME=str(ROOT/'cache'),TMPDIR=str(ROOT/'tmp'))
    for task,cmd in commands:
        with (OUT/f'{name}-{task}.log').open('x') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
            status.update(status='running',task=f'{name}-{task}',child_pid=child.pid);save()
            while child.poll() is None:time.sleep(5);save()
            status['completed'].append({'task':task,'exit_code':child.returncode,'finished_at':stamp()});save()
            if child.returncode:status['status']='failed';save();return
    summary=OUT/'evaluation'/f'validation-{name}'/'status.json'
    status['status']='completed' if json.loads(summary.read_text())['status']=='completed' else 'incomplete_validation';save()
if __name__=='__main__':main()

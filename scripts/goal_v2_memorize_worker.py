"""Training-fit feasibility probe plus frozen-start controls. No sealed-test access."""
import argparse,datetime,fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',choices=['0','2','3'],required=True);a=ap.parse_args()
    lock=(OUT/f'worker-memorize-{a.gpu}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=OUT/f'worker-memorize-{a.gpu}.json';assert not path.exists()
    state={'pid':os.getpid(),'gpu':a.gpu,'status':'waiting','started_at':stamp(),'completed':[]};start=time.monotonic()
    def save():
        state['updated_at']=stamp();t=path.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(path)
    save();prior=OUT/('worker-trial-json-calibrated-lr1e-4-s20261103.json' if a.gpu=='0' else f'worker-json-{a.gpu}.json')
    while True:
        previous=json.loads(prior.read_text())
        if previous['status']=='failed':state['status']='incomplete_previous_queue_failed';save();return
        if previous['status']=='completed':
            used=int(subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
            if used<512:break
        if time.monotonic()-start>14400:state['status']='observation_time_limit';save();return
        time.sleep(10)
    py=str(ROOT/'envs/iea-rl/bin/python');commands=[]
    if a.gpu=='0':
        name='json-memorize64-lr1e-4-s20261103';adapter=OUT/name/'final'
        commands.append(('train',[py,str(ROOT/'scripts/goal_v2_train_memorize64.py'),'--name',name,'--mode','parameters','--lr','1e-4','--gpu',a.gpu,'--training-file','train-memorize64.json']))
    else:
        base='A' if a.gpu=='3' else 'B';name=f'{base}-json-memorize64-control'
        adapter=ROOT/('outputs/stage1-A-image-edit-3k-v1' if base=='A' else 'outputs/stage1-B-image-edit-3k-v1')
    commands.append(('training-probe',[py,str(ROOT/'scripts/goal_v2_evaluate_memorize64.py'),'--adapter',str(adapter),'--name',name,'--split','train_probe','--gpu',a.gpu,'--max-gpu-seconds','3600']))
    if a.gpu=='0':commands.append(('validate',[py,str(ROOT/'scripts/goal_v2_evaluate_json.py'),'--adapter',str(adapter),'--name',name,'--split','validation','--gpu',a.gpu,'--max-gpu-seconds','5400']))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,PYTHONPATH=str(ROOT/'scripts'),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',XDG_CACHE_HOME=str(ROOT/'cache'),TMPDIR=str(ROOT/'tmp'))
    for task,cmd in commands:
        with (OUT/f'{name}-{task}.log').open('x') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
            state.update(status='running',task=f'{name}-{task}',child_pid=child.pid);save()
            while child.poll() is None:time.sleep(5);save()
            state['completed'].append({'task':task,'exit_code':child.returncode,'finished_at':stamp()});save()
            if child.returncode:state['status']='failed';save();return
    p=OUT/('evaluation' if a.gpu=='0' else 'training-probe-evaluation')/(f'validation-{name}' if a.gpu=='0' else f'train_probe-{name}')/'status.json'
    state['status']='completed' if json.loads(p.read_text())['status']=='completed' else 'incomplete_evaluation';save()
if __name__=='__main__':main()

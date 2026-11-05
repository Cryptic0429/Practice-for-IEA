"""Small JSON-only format trial with frozen A/B prompt controls, after GPU handoff."""
import argparse,datetime,fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',choices=['0','2','3'],required=True);a=ap.parse_args()
    lock=(OUT/f'worker-json-{a.gpu}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=OUT/f'worker-json-{a.gpu}.json';assert not path.exists()
    status={'pid':os.getpid(),'gpu':a.gpu,'status':'waiting','started_at':stamp(),'completed':[]};start=time.monotonic()
    def save():
        status['updated_at']=stamp();t=path.with_suffix('.tmp');t.write_text(json.dumps(status,indent=2)+'\n');t.replace(path)
    save();parent=OUT/(f'worker-calibrated-{a.gpu}.json' if a.gpu in ['0','3'] else 'worker-2.json')
    while True:
        previous=json.loads(parent.read_text())
        if previous['status']=='failed':status['status']='incomplete_previous_queue_failed';save();return
        if previous['status']=='completed':
            memory=int(subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
            if memory<512:break
        if time.monotonic()-start>14400:status['status']='observation_time_limit';save();return
        time.sleep(10)
    py=str(ROOT/'envs/iea-rl/bin/python');commands=[]
    if a.gpu=='0':
        name='json-calibrated-lr3e-5-s20261103';adapter=OUT/name/'final'
        commands.append(('train',[py,str(ROOT/'scripts/goal_v2_train_json.py'),'--name',name,'--mode','parameters','--lr','3e-5','--gpu',a.gpu,'--training-file','train-contrast-calibrated.json']))
    else:
        base='A' if a.gpu=='2' else 'B';name=f'{base}-json-schema'
        adapter=ROOT/('outputs/stage1-A-image-edit-3k-v1' if base=='A' else 'outputs/stage1-B-image-edit-3k-v1')
    commands.append(('validate',[py,str(ROOT/'scripts/goal_v2_evaluate_json.py'),'--adapter',str(adapter),'--name',name,'--split','validation','--gpu',a.gpu,'--max-gpu-seconds','5400']))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,PYTHONPATH=str(ROOT/'scripts'),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',XDG_CACHE_HOME=str(ROOT/'cache'),TMPDIR=str(ROOT/'tmp'))
    for task,command in commands:
        with (OUT/f'{name}-{task}.log').open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
            status.update(status='running',task=f'{name}-{task}',child_pid=child.pid);save()
            while child.poll() is None:time.sleep(5);save()
            status['completed'].append({'task':task,'exit_code':child.returncode,'finished_at':stamp()});save()
            if child.returncode:status['status']='failed';save();return
    state=json.loads((OUT/'evaluation'/f'validation-{name}'/'status.json').read_text())
    status['status']='completed' if state['status']=='completed' else 'incomplete_validation';save()
if __name__=='__main__':main()

"""Durable per-GPU queue; stop on errors, never restart or overwrite a run."""
import argparse, datetime, fcntl, json, os, subprocess, time
from pathlib import Path
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',choices=['0','2','3'],required=True);a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/f'worker-{a.gpu}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    py=str(ROOT/'envs/iea-rl/bin/python');scripts=ROOT/'scripts';tasks=[]
    models={'A':ROOT/'outputs/stage1-A-image-edit-3k-v1','B':ROOT/'outputs/stage1-B-image-edit-3k-v1',
            'SFT':ROOT/'outputs/stage2-A-edit-grpo-v1/sft-control/final','GRPO':ROOT/'outputs/stage2-A-edit-grpo-v1/formal-v1/selected'}
    def evaluate(name,path):
        return [py,str(scripts/'goal_v2_evaluate.py'),'--adapter',str(path),'--name',name,'--split','validation','--gpu',a.gpu,'--max-gpu-seconds','5400']
    if a.gpu in ['0','3']:
        for name in (['A','SFT'] if a.gpu=='0' else ['B','GRPO']):tasks.append((f'eval-{name}',evaluate(name,models[name])))
    else:
        for name,mode,lr in [('full-lr1e-5-s20261103','full','1e-5'),('parameters-lr1e-5-s20261103','parameters','1e-5'),('parameters-lr3e-5-s20261103','parameters','3e-5')]:
            tasks.append((name,[py,str(scripts/'goal_v2_train.py'),'--name',name,'--mode',mode,'--lr',lr,'--gpu',a.gpu]))
        for name in ['full-lr1e-5-s20261103','parameters-lr1e-5-s20261103','parameters-lr3e-5-s20261103']:
            tasks.append((f'eval-{name}',evaluate(name,OUT/name/'final')))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,PYTHONPATH=str(scripts),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',
             XDG_CACHE_HOME=str(ROOT/'cache'),TMPDIR=str(ROOT/'tmp'))
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    status={'pid':os.getpid(),'gpu':a.gpu,'started_at':stamp(),'status':'running','completed':[],'tasks':tasks}
    def save():
        status['updated_at']=stamp();p=OUT/f'worker-{a.gpu}.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(status,indent=2)+'\n');t.replace(p)
    save()
    for name,cmd in tasks:
        with (OUT/f'{name}.log').open('x') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
            status.update(task=name,child_pid=child.pid,task_started_at=stamp());save()
            while child.poll() is None:time.sleep(5);save()
            status['completed'].append({'task':name,'exit_code':child.returncode,'finished_at':stamp()});save()
            if child.returncode:
                status['status']='failed';save();return
    status['status']='completed';save()
if __name__=='__main__': main()

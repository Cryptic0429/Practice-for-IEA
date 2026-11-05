"""Validate only fully saved checkpoints from the bounded full512 long training job."""
import argparse,datetime,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--gpu',choices=['0','2','3'],required=True);a=ap.parse_args()
    path=OUT/'worker-long3k-validation.json';assert not path.exists();source=OUT/a.name;start=time.monotonic()
    state={'pid':os.getpid(),'status':'waiting_checkpoint','gpu':a.gpu,'source':str(source),'completed':[]}
    def write():
        state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();t=path.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(path)
    write();py=str(ROOT/'envs/iea-rl/bin/python');env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,PYTHONPATH=str(ROOT/'scripts'),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',XDG_CACHE_HOME=str(ROOT/'cache'),TMPDIR=str(ROOT/'tmp'))
    for step in [256,512,1024]:
        cp=source/f'checkpoint-{step}'
        while True:
            status_path=source/'status.json';status=json.loads(status_path.read_text()) if status_path.exists() else {}
            ready=status.get('step',0)>step or status.get('status')=='completed'
            files=['adapter_model.safetensors','optimizer.pt','trainer_state.json','rng_state.pth','scheduler.pt']
            ready=ready and all((cp/p).exists() for p in files)
            if ready and a.gpu=='0':
                previous=json.loads((OUT/'json-calibrated-long1024-lr1e-4-s20261103/status.json').read_text())
                ready=previous['status'] in ['completed','failed','time_limit_stopped']
            if ready:
                used=int(subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
                if used<512:break
            launch=json.loads((OUT/'long3k-training-launch.json').read_text())
            live=Path(f'/proc/{launch["pid"]}').exists()
            if not ready and (status.get('status') in ['failed','time_limit_stopped'] or (not live and status.get('status')!='completed')):
                state.update(status='training_unavailable',training_state=status);write();return
            if time.monotonic()-start>18000:state['status']='observation_limit';write();return
            time.sleep(5)
        audited=subprocess.run([py,str(ROOT/'scripts/goal_v2_checkpoint_audit.py'),str(cp),'--step',str(step)],capture_output=True,text=True)
        if audited.returncode:state.update(status='checkpoint_audit_failed',step=step,error=audited.stderr);write();return
        (source/f'checkpoint-{step}-audit.json').write_text(audited.stdout)
        name=f'{a.name}-step{step}'
        with (OUT/f'{name}-validate.log').open('x') as log:
            child=subprocess.Popen([py,str(ROOT/'scripts/goal_v2_evaluate_json.py'),'--adapter',str(cp),'--name',name,'--split','validation','--gpu',a.gpu,'--max-gpu-seconds','5400'],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='running',task=name,step=step,child_pid=child.pid);write()
            while child.poll() is None:time.sleep(5);write()
            if child.returncode:state.update(status='validation_failed',exit_code=child.returncode);write();return
        result=OUT/'evaluation'/f'validation-{name}'/'status.json'
        if json.loads(result.read_text())['status']!='completed':state['status']='incomplete_validation';write();return
        state['completed'].append({'step':step,'name':name});state.update(status='waiting_checkpoint');write()
    state['status']='completed';write()
if __name__=='__main__':main()

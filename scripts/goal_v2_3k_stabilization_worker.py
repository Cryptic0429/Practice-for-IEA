"""Primary-seed-only fixed checkpoint averaging control, before replica validation."""
import datetime,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def main():
    path=OUT/'worker-3k-stabilization.json';assert not path.exists();state={'pid':os.getpid(),'gpu':'2','status':'waiting_gpu_handoff','completed':[]};start=time.monotonic()
    def save():
        state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
    save()
    while True:
        parent=json.loads((OUT/'worker-replica-challenge-20261104.json').read_text())
        if parent['status']=='completed':
            if int(subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))<512:break
        if parent['status'] in ['failed','checkpoint_audit_failed','checkpoint_final_identity_failed','replica_unavailable']:state.update(status='dependency_failed',parent=parent);save();return
        if time.monotonic()-start>7200:state['status']='observation_limit';save();return
        time.sleep(5)
    name='json3k-primary-swa512-1024-s20261103';source=OUT/'json-calibrated3k-long1024-lr1e-4-s20261103';adapter=OUT/name/'adapter';py=str(ROOT/'envs/iea-rl/bin/python')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',PYTHONPATH=str(ROOT/'scripts'),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',XDG_CACHE_HOME=str(ROOT/'cache'),TMPDIR=str(ROOT/'tmp'))
    commands=[('average',[py,str(ROOT/'scripts/goal_v2_average_adapter.py'),'--left',str(source/'checkpoint-512'),'--right',str(source/'checkpoint-1024'),'--output',str(adapter)]),
              ('validate',[py,str(ROOT/'scripts/goal_v2_evaluate_json.py'),'--adapter',str(adapter),'--name',name,'--split','validation','--gpu','2','--max-gpu-seconds','5400']),
              ('challenge',[py,str(ROOT/'scripts/goal_v2_instruction_challenge.py'),'--adapter',str(adapter),'--name',name,'--gpu','2','--format','json'])]
    for task,cmd in commands:
        with (OUT/f'{name}-{task}.log').open('x') as log:
            child=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);state.update(status='running',task=task,name=name,child_pid=child.pid);save()
            while child.poll() is None:time.sleep(5);save()
            state['completed'].append({'task':task,'exit_code':child.returncode});save()
            if child.returncode:state['status']='failed';save();return
    state['status']='completed';save()
if __name__=='__main__':main()

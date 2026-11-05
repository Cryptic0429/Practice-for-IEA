"""Use GPU3 only after its locked final evaluations finish, for a training-state restore check."""
import datetime,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def main():
    path=OUT/'worker-restore-smoke.json';assert not path.exists();state={'pid':os.getpid(),'gpu':'3','status':'waiting_final_gpu3'};start=time.monotonic()
    def save():
        state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
    save()
    while True:
        previous=json.loads((OUT/'worker-final-3.json').read_text())
        if previous['status']=='completed' and int(subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))<512:break
        if previous['status'] in ['failed','incomplete_model','handoff_observation_limit']:state.update(status='dependency_failed',previous=previous);save();return
        if time.monotonic()-start>14400:state['status']='observation_limit';save();return
        time.sleep(5)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHONPATH=str(ROOT/'scripts'),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',XDG_CACHE_HOME=str(ROOT/'cache'),TMPDIR=str(ROOT/'tmp'))
    with (OUT/'checkpoint-restore-smoke.log').open('x') as log:
        child=subprocess.Popen([str(ROOT/'envs/iea-rl/bin/python'),str(ROOT/'scripts/goal_v2_restore_smoke.py'),'--gpu','3'],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        state.update(status='running',child_pid=child.pid);save()
        while child.poll() is None:time.sleep(5);save()
        state['status']='completed' if child.returncode==0 else 'failed';state['exit_code']=child.returncode;save()
if __name__=='__main__':main()

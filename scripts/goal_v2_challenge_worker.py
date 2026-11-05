"""Queue balanced instruction checks after the existing GPU3 validation worker."""
import datetime,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    path=OUT/'worker-instruction-challenge.json';assert not path.exists();state={'pid':os.getpid(),'gpu':'3','status':'waiting_previous_validation','completed':[]};start=time.monotonic()
    def save():
        state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
    save()
    while True:
        previous=json.loads((OUT/'worker-long512-validation.json').read_text())
        if previous['status']=='completed':break
        if previous['status'] in ['validation_failed','checkpoint_audit_failed','training_unavailable','incomplete_validation']:
            state.update(status='dependency_failed',previous=previous);save();return
        if time.monotonic()-start>14400:state['status']='observation_limit';save();return
        time.sleep(5)
    py=str(ROOT/'envs/iea-rl/bin/python');env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHONPATH=str(ROOT/'scripts'),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',XDG_CACHE_HOME=str(ROOT/'cache'),TMPDIR=str(ROOT/'tmp'))
    for kind in ['B','candidate']:
        if kind=='B':adapter=ROOT/'outputs/stage1-B-image-edit-3k-v1';name='B-original-format';mode='original'
        else:
            lock=OUT/'development-model-lock.json';state['status']='waiting_candidate_development_lock';save()
            while not lock.exists():
                if time.monotonic()-start>14400:state['status']='candidate_lock_observation_limit';save();return
                time.sleep(5)
            locked=json.loads(lock.read_text());assert locked['status']=='frozen_development_candidate'
            adapter=Path(locked['adapter']);assert sha(adapter/'adapter_model.safetensors')==locked['adapter_sha256'];name=locked['name'];mode=locked['format']
        while int(subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))>=512:time.sleep(5)
        with (OUT/f'instruction-challenge-{kind}.log').open('x') as log:
            child=subprocess.Popen([py,str(ROOT/'scripts/goal_v2_instruction_challenge.py'),'--adapter',str(adapter),'--name',name,'--gpu','3','--format',mode],stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
            state.update(status='running',task=name,child_pid=child.pid);save()
            while child.poll() is None:time.sleep(5);save()
            state['completed'].append({'name':name,'exit_code':child.returncode});save()
            if child.returncode:state['status']='failed';save();return
    state['status']='completed';save()
if __name__=='__main__':main()

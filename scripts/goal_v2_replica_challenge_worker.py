"""Audit each independent replica and repeat the same frozen instruction challenge."""
import argparse,datetime,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',choices=['2','3'],required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args()
    assert (a.gpu,a.seed) in [('2',20261104),('3',20261105)]
    path=OUT/f'worker-replica-challenge-{a.seed}.json';assert not path.exists();state={'pid':os.getpid(),'gpu':a.gpu,'seed':a.seed,'status':'waiting_replica_validation'};start=time.monotonic()
    def save():
        state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();t=path.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(path)
    save()
    while True:
        parent=json.loads((OUT/f'worker-replication-{a.seed}.json').read_text())
        if parent['status']=='completed':break
        if parent['status'] in ['failed','instruction_guardrail_failed','instruction_check_unavailable','instruction_observation_limit','gpu_handoff_observation_limit']:
            state.update(status='replica_unavailable',details=parent);save();return
        if time.monotonic()-start>21600:state['status']='observation_limit';save();return
        time.sleep(5)
    name=parent['name'];run=OUT/name;py=str(ROOT/'envs/iea-rl/bin/python')
    audited=subprocess.run([py,str(ROOT/'scripts/goal_v2_checkpoint_audit.py'),str(run/'checkpoint-256'),'--step','256'],text=True,capture_output=True)
    if audited.returncode:state.update(status='checkpoint_audit_failed',error=audited.stderr);save();return
    (run/'checkpoint-256-audit.json').write_text(audited.stdout)
    verify='''import sys,torch,json;from safetensors.torch import load_file;from pathlib import Path
p=Path(sys.argv[1]);a=load_file(str(p/'checkpoint-256/adapter_model.safetensors'));b=load_file(str(p/'final/adapter_model.safetensors'));assert a.keys()==b.keys() and len(a)==112;assert all(torch.equal(a[k],b[k]) for k in a);(p/'checkpoint-final-tensor-identity.json').write_text(json.dumps({'passed':True,'tensors':112,'step':256})+'\\n')'''
    result=subprocess.run([py,'-c',verify,str(run)],capture_output=True,text=True)
    if result.returncode:state.update(status='checkpoint_final_identity_failed',error=result.stderr);save();return
    while int(subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))>=512:time.sleep(5)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,PYTHONPATH=str(ROOT/'scripts'),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',XDG_CACHE_HOME=str(ROOT/'cache'),TMPDIR=str(ROOT/'tmp'))
    with (OUT/f'instruction-challenge-replica-{a.seed}.log').open('x') as log:
        child=subprocess.Popen([py,str(ROOT/'scripts/goal_v2_instruction_challenge.py'),'--adapter',str(run/'final'),'--name',name,'--gpu',a.gpu,'--format','json'],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        state.update(status='running',task=name,child_pid=child.pid);save()
        while child.poll() is None:time.sleep(5);save()
        state['status']='completed' if child.returncode==0 else 'failed';state['exit_code']=child.returncode;save()
if __name__=='__main__':main()

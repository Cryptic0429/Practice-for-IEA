"""Independent full3k training; primary-seed-only export selection precedes replica validation."""
import argparse,datetime,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',choices=['0','3'],required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args();assert (a.gpu,a.seed) in [('0',20261104),('3',20261105)]
    path=OUT/f'worker-3k-replication-{a.seed}.json';assert not path.exists();state={'pid':os.getpid(),'gpu':a.gpu,'seed':a.seed,'status':'checking_lock','completed':[]};start=time.monotonic()
    def save():
        state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();t=path.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(path)
    save();lock=json.loads((OUT/'development-training-lock-3k-v2.json').read_text());assert lock['status']=='frozen_training_recipe' and lock['instruction_guardrail_passed']
    assert sha(ROOT/'data/goal-v2-20261103/train3k-contrast-calibrated.json')==lock['training_sha256'];assert sha(ROOT/'scripts/goal_v2_train_json3k_long.py')==lock['training_code_sha256']
    while int(subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))>=512:
        if time.monotonic()-start>14400:state['status']='gpu_observation_limit';save();return
        time.sleep(5)
    py=str(ROOT/'envs/iea-rl/bin/python');env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,PYTHONPATH=str(ROOT/'scripts'),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',XDG_CACHE_HOME=str(ROOT/'cache'),TMPDIR=str(ROOT/'tmp'))
    name=f'json3k-replica1024-lr1e-4-s{a.seed}';run=OUT/name
    def execute(task,cmd):
        with (OUT/f'{name}-{task}.log').open('x') as log:
            child=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);state.update(status='running',task=task,child_pid=child.pid,name=name);save()
            while child.poll() is None:time.sleep(5);save()
            state['completed'].append({'task':task,'exit_code':child.returncode});save()
            if child.returncode:state.update(status='failed',failed_task=task);save();return False
        return True
    if not execute('train',[py,str(ROOT/'scripts/goal_v2_train_json3k_long.py'),'--name',name,'--mode','parameters','--lr','1e-4','--gpu',a.gpu,'--seed',str(a.seed),'--training-file','train3k-contrast-calibrated.json']):return
    audited=subprocess.run([py,str(ROOT/'scripts/goal_v2_checkpoint_audit.py'),str(run/'checkpoint-1024'),'--step','1024'],capture_output=True,text=True)
    if audited.returncode:state.update(status='checkpoint_audit_failed',error=audited.stderr);save();return
    (run/'checkpoint-1024-audit.json').write_text(audited.stdout)
    selection=OUT/'3k-export-selection.json';state['status']='waiting_primary_export_selection';save()
    while not selection.exists():
        if time.monotonic()-start>21600:state['status']='export_selection_observation_limit';save();return
        time.sleep(5)
    selected=json.loads(selection.read_text());assert selected['status']=='frozen_before_replica_validation' and selected['variant'] in ['raw','swa']
    adapter=run/'final';evaluation_name=name+'-'+selected['variant']
    if selected['variant']=='swa':
        adapter=run/'swa'
        if not execute('average',[py,str(ROOT/'scripts/goal_v2_average_adapter.py'),'--left',str(run/'checkpoint-512'),'--right',str(run/'checkpoint-1024'),'--output',str(adapter)]):return
    state.update(evaluation_name=evaluation_name,adapter=str(adapter),export_selection_sha256=sha(selection));save()
    if not execute('validate',[py,str(ROOT/'scripts/goal_v2_evaluate_json.py'),'--adapter',str(adapter),'--name',evaluation_name,'--split','validation','--gpu',a.gpu,'--max-gpu-seconds','5400']):return
    status=json.loads((OUT/'evaluation'/f'validation-{evaluation_name}'/'status.json').read_text())
    if status['status']!='completed' or status['completed']!=200:state['status']='incomplete_validation';save();return
    if not execute('challenge',[py,str(ROOT/'scripts/goal_v2_instruction_challenge.py'),'--adapter',str(adapter),'--name',evaluation_name,'--gpu',a.gpu,'--format','json']):return
    state['status']='completed';save()
if __name__=='__main__':main()

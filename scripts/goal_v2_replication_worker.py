"""Independent seeds of locked256-update recipe, gated on instruction checks and GPU handoff."""
import argparse,datetime,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',choices=['2','3'],required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args()
    assert (a.gpu,a.seed) in [('2',20261104),('3',20261105)]
    path=OUT/f'worker-replication-{a.seed}.json';assert not path.exists();start=time.monotonic();state={'pid':os.getpid(),'gpu':a.gpu,'seed':a.seed,'status':'waiting_instruction_guardrail'}
    def save():
        state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();t=path.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(path)
    save();lock=json.loads((OUT/'development-model-lock.json').read_text());assert lock['status']=='frozen_development_candidate' and a.seed in lock['seed_list']
    while True:
        challenge=json.loads((OUT/'worker-instruction-challenge.json').read_text())
        if challenge['status']=='completed':break
        if challenge['status'] in ['failed','dependency_failed','observation_limit','candidate_lock_observation_limit']:
            state.update(status='instruction_check_unavailable',details=challenge);save();return
        if time.monotonic()-start>18000:state['status']='instruction_observation_limit';save();return
        time.sleep(5)
    baseline=json.loads((OUT/'instruction-challenge/B-original-format/summary.json').read_text());candidate=json.loads((OUT/'instruction-challenge'/lock['name']/'summary.json').read_text())
    assert baseline['requests']==candidate['requests']==48 and baseline['source_groups']==candidate['source_groups']==8
    fields=['executable','aligned_requests','strict_pairs'];passed=all(candidate[k]>=baseline[k] for k in fields)
    gate={'passed':passed,'rule':'Maintain B execution, aligned requests and pairs with both requested directions aligned; fixed before challenge inference. Ordered pairs reported separately.','baseline':{k:baseline[k] for k in fields+['ordered_pairs']},'candidate':{k:candidate[k] for k in fields+['ordered_pairs']},'baseline_summary_sha256':sha(OUT/'instruction-challenge/B-original-format/summary.json'),'candidate_summary_sha256':sha(OUT/'instruction-challenge'/lock['name']/'summary.json'),'development_lock_sha256':sha(OUT/'development-model-lock.json')}
    (OUT/f'replication-instruction-gate-{a.seed}.json').write_text(json.dumps(gate,indent=2)+'\n')
    if not passed:state.update(status='instruction_guardrail_failed',gate=gate);save();return
    state['status']='waiting_gpu_handoff';save()
    while True:
        parent=json.loads((OUT/('worker-3k.json' if a.gpu=='2' else 'worker-instruction-challenge.json')).read_text())
        terminal=parent['status'] in (['training_completed_validation_separate','failed'] if a.gpu=='2' else ['completed'])
        if terminal:
            used=int(subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
            if used<512:break
        if time.monotonic()-start>21600:state['status']='gpu_handoff_observation_limit';save();return
        time.sleep(5)
    assert lock['selected_step']==256 and lock['scheduler_horizon']==1024 and lock['warmup_steps']==52
    assert sha(ROOT/'data/goal-v2-20261103'/lock['training_file'])==lock['training_sha256']
    assert sha(ROOT/'scripts/goal_v2_json_runtime.py')==lock['evaluation_identity']['goal_v2_json_runtime.py']
    name=f'json-calibrated-locked256-lr1e-4-s{a.seed}';py=str(ROOT/'envs/iea-rl/bin/python')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,PYTHONPATH=str(ROOT/'scripts'),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',XDG_CACHE_HOME=str(ROOT/'cache'),TMPDIR=str(ROOT/'tmp'))
    commands=[('train',[py,str(ROOT/'scripts/goal_v2_train_json_replica.py'),'--name',name,'--mode','parameters','--lr','1e-4','--gpu',a.gpu,'--seed',str(a.seed),'--training-file',lock['training_file']]),
              ('validate',[py,str(ROOT/'scripts/goal_v2_evaluate_json.py'),'--adapter',str(OUT/name/'final'),'--name',name,'--split','validation','--gpu',a.gpu,'--max-gpu-seconds','5400'])]
    state['completed']=[]
    for task,cmd in commands:
        with (OUT/f'{name}-{task}.log').open('x') as log:
            child=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);state.update(status='running',task=task,name=name,child_pid=child.pid);save()
            while child.poll() is None:time.sleep(5);save()
            state['completed'].append({'task':task,'exit_code':child.returncode});save()
            if child.returncode:state['status']='failed';save();return
    state['status']='completed';save()
if __name__=='__main__':main()

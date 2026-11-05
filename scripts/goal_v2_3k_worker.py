"""Broader clean training pool: CPU ridge diagnostic then matched-budget JSON SFT."""
import datetime,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def main():
    path=OUT/'worker-3k.json';assert not path.exists();start=time.monotonic();state={'pid':os.getpid(),'status':'waiting_data','completed':[]}
    def save():
        state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();t=path.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(path)
    save()
    for name,directory in [('calibration3k','contrast-label-search3k'),('feature-extraction3k','frozen-A-features3k-v1')]:
        while True:
            launch=json.loads((OUT/f'{name}-launch.json').read_text());live=Path(f'/proc/{launch["pid"]}').exists();status=OUT/directory/'status.json'
            s=json.loads(status.read_text()) if status.exists() else {}
            if s.get('status')=='completed':break
            if s.get('status')=='failed' or not live:state.update(status='prerequisite_failed',prerequisite=name,details=s);save();return
            if time.monotonic()-start>7200:state['status']='observation_limit';save();return
            time.sleep(5)
    summary=json.loads((OUT/'contrast-label-search3k/summary.json').read_text())
    if not (summary['new_mean_l1']<.99*summary['old_mean_l1'] and summary['changed_labels']>=25):state.update(status='label_quality_gate_not_met',summary=summary);save();return
    py=str(ROOT/'envs/iea-rl/bin/python');env=dict(os.environ,PYTHONPATH=str(ROOT/'scripts'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',XDG_CACHE_HOME=str(ROOT/'cache'),TMPDIR=str(ROOT/'tmp'))
    jobs=[('ridge-train',[py,str(ROOT/'scripts/goal_v2_train_numeric_ridge3k.py')]),
          ('ridge-validate',[py,str(ROOT/'scripts/goal_v2_evaluate_numerical_head.py'),'--name','numeric-ridge3k-cv-s20261103','--split','validation'])]
    for task,cmd in jobs:
        with (OUT/f'3k-{task}.log').open('x') as log:
            child=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='running',task=task,child_pid=child.pid);save()
            while child.poll() is None:time.sleep(5);save()
            state['completed'].append({'task':task,'exit_code':child.returncode});save()
            if child.returncode:state['status']='failed';save();return
    while int(subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))>=512:
        if time.monotonic()-start>14400:state['status']='gpu_observation_limit';save();return
        time.sleep(5)
    name='json-calibrated3k-long1024-lr1e-4-s20261103';env['CUDA_VISIBLE_DEVICES']='2'
    with (OUT/f'{name}-train.log').open('x') as log:
        child=subprocess.Popen([py,str(ROOT/'scripts/goal_v2_train_json3k_long.py'),'--name',name,'--mode','parameters','--lr','1e-4','--gpu','2','--training-file','train3k-contrast-calibrated.json'],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        state.update(status='running',task='json3k-train',gpu='2',child_pid=child.pid);save()
        launch=OUT/'long3k-training-launch.json';assert not launch.exists();launch.write_text(json.dumps({'pid':child.pid,'gpu':2,'name':name,'steps':1024,'rows':3000,'groups':1078,'matched_512_budget':True},indent=2)+'\n')
        with (OUT/'3k-validation-worker.log').open('x') as vlog:
            validator=subprocess.Popen([py,str(ROOT/'scripts/goal_v2_3k_validation_worker.py'),'--name',name,'--gpu','0'],env=env,cwd=ROOT,stdout=vlog,stderr=subprocess.STDOUT,start_new_session=True)
        state['validation_worker_pid']=validator.pid;save()
        while child.poll() is None:time.sleep(5);save()
        state['completed'].append({'task':'json3k-train','exit_code':child.returncode});state['status']='training_completed_validation_separate' if child.returncode==0 else 'failed';save()
if __name__=='__main__':main()

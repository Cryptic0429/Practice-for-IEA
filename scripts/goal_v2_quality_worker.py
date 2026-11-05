"""One bounded quality-weighted3k control on GPU2, only after train-only audit warrants it."""
import datetime,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def main():
    path=OUT/'worker-qualityweighted3k.json';assert not path.exists();state={'pid':os.getpid(),'gpu':'2','status':'waiting_audit','completed':[]};start=time.monotonic()
    def save():
        state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
    save()
    while True:
        p=OUT/'calibrated3k-quality-audit/status.json';audit=json.loads(p.read_text()) if p.exists() else {}
        if audit.get('status')=='completed':break
        launch=json.loads((OUT/'calibrated3k-quality-audit-launch.json').read_text())
        if audit.get('status')=='failed' or not Path('/proc',str(launch['pid'])).exists():state.update(status='audit_unavailable',details=audit);save();return
        if time.monotonic()-start>7200:state['status']='observation_limit';save();return
        time.sleep(5)
    report=json.loads((OUT/'calibrated3k-quality-audit/summary.json').read_text())
    if report['labels_worsening_original']<.1*report['training_rows'] and report['clear_direction_oppositions']<.05*max(1,report['direction_assessable']):state.update(status='quality_weighting_not_warranted',summary=report);save();return
    while True:
        prior=json.loads((OUT/'worker-3k-stabilization.json').read_text())
        if prior['status']=='completed' and int(subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))<512:break
        if time.monotonic()-start>14400:state['status']='gpu_observation_limit';save();return
        time.sleep(5)
    py=str(ROOT/'envs/iea-rl/bin/python');name='json3k-qualityweighted512-lr1e-4-s20261103';env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',PYTHONPATH=str(ROOT/'scripts'),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',XDG_CACHE_HOME=str(ROOT/'cache'),TMPDIR=str(ROOT/'tmp'))
    commands=[('train',[py,str(ROOT/'scripts/goal_v2_train_json3k_weighted.py'),'--name',name,'--mode','parameters','--lr','1e-4','--gpu','2','--training-file','train3k-contrast-calibrated.json']),
              ('validate',[py,str(ROOT/'scripts/goal_v2_evaluate_json.py'),'--adapter',str(OUT/name/'final'),'--name',name,'--split','validation','--gpu','2','--max-gpu-seconds','5400']),
              ('challenge',[py,str(ROOT/'scripts/goal_v2_instruction_challenge.py'),'--adapter',str(OUT/name/'final'),'--name',name,'--gpu','2','--format','json'])]
    for task,cmd in commands:
        with (OUT/f'{name}-{task}.log').open('x') as log:
            child=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);state.update(status='running',task=task,name=name,child_pid=child.pid);save()
            while child.poll() is None:time.sleep(5);save()
            state['completed'].append({'task':task,'exit_code':child.returncode});save()
            if child.returncode:state['status']='failed';save();return
    state['status']='completed';save()
if __name__=='__main__':main()

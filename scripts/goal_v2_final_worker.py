"""Run locked71-model evaluations after explicit same-GPU task handoff; no model changes."""
import argparse,datetime,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',choices=['0','2','3'],required=True);a=ap.parse_args()
    lock=json.loads((OUT/'final-models-lock.json').read_text());assert lock['status']=='frozen_for_final_test'
    assigned={'0':['A','SFT','Seed20261103'],'2':['B','Seed20261104'],'3':['GRPO','Seed20261105']}[a.gpu]
    assert all(n in lock['models'] for n in assigned)
    path=OUT/f'worker-final-{a.gpu}.json';assert not path.exists();state={'pid':os.getpid(),'gpu':a.gpu,'status':'waiting_gpu_handoff','assigned':assigned,'completed':[]};start=time.monotonic()
    def save():
        state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
    save()
    parent_file=OUT/({'0':'worker-3k-replication-20261104.json','3':'worker-3k-replication-20261105.json','2':'worker-qualityweighted3k.json'}[a.gpu])
    while True:
        parent=json.loads(parent_file.read_text())
        allowed=['completed'] if a.gpu!='2' else ['completed','quality_weighting_not_warranted','failed','audit_unavailable','observation_limit','gpu_observation_limit']
        if parent['status'] in allowed:
            used=int(subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
            if used<512:break
        if time.monotonic()-start>14400:state['status']='handoff_observation_limit';save();return
        time.sleep(5)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,PYTHONPATH=str(ROOT/'scripts'),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',XDG_CACHE_HOME=str(ROOT/'cache'),TMPDIR=str(ROOT/'tmp'));py=str(ROOT/'envs/iea-rl/bin/python')
    for name in assigned:
        with (OUT/f'final-{name}.log').open('x') as log:
            child=subprocess.Popen([py,str(ROOT/'scripts/goal_v2_final_evaluate.py'),'--model',name,'--gpu',a.gpu,'--max-gpu-seconds','5400'],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='running',task=name,child_pid=child.pid);save()
            while child.poll() is None:time.sleep(5);save()
            state['completed'].append({'model':name,'exit_code':child.returncode});save()
            if child.returncode:state['status']='failed';save();return
        result=json.loads((OUT/'final-evaluation'/name/'status.json').read_text())
        if result['status']!='completed' or result['completed']!=71:state.update(status='incomplete_model',details=result);save();return
    state['status']='completed';save()
if __name__=='__main__':main()

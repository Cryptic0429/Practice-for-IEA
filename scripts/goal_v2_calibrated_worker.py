"""Bounded small label-calibration pilots after existing GPU queues finish."""
import argparse,datetime,fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',choices=['0','3'],required=True);a=ap.parse_args()
    lock=(OUT/f'worker-calibrated-{a.gpu}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    statepath=OUT/f'worker-calibrated-{a.gpu}.json';assert not statepath.exists()
    status={'pid':os.getpid(),'gpu':a.gpu,'status':'waiting','started_at':stamp(),'completed':[]};start=time.monotonic()
    def save():
        status['updated_at']=stamp();tmp=statepath.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(statepath)
    save()
    while True:
        previous=json.loads((OUT/f'worker-{a.gpu}.json').read_text());search=OUT/'contrast-label-search/summary.json'
        if previous['status']=='failed':status.update(status='blocked_previous_queue_failed');save();return
        if previous['status']=='completed' and search.exists():
            report=json.loads(search.read_text());assert report['status']=='completed'
            if report['new_mean_l1']>=.99*report['old_mean_l1'] or report['changed_labels']<25:
                status.update(status='skipped_insufficient_training_label_gain',search_report=report);save();return
            memory=int(subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
            if memory<512:break
        if time.monotonic()-start>14400:status['status']='observation_time_limit';save();return
        time.sleep(15)
    lr='1e-5' if a.gpu=='0' else '3e-5';name=f'contrast-calibrated-lr{lr}-s20261103';py=str(ROOT/'envs/iea-rl/bin/python')
    commands=[('train',[py,str(ROOT/'scripts/goal_v2_train.py'),'--name',name,'--mode','parameters','--lr',lr,'--gpu',a.gpu,'--training-file','train-contrast-calibrated.json']),
              ('validate',[py,str(ROOT/'scripts/goal_v2_evaluate.py'),'--adapter',str(OUT/name/'final'),'--name',name,'--split','validation','--gpu',a.gpu,'--max-gpu-seconds','5400'])]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,PYTHONPATH=str(ROOT/'scripts'),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',XDG_CACHE_HOME=str(ROOT/'cache'),TMPDIR=str(ROOT/'tmp'))
    for task,command in commands:
        with (OUT/f'{name}-{task}.log').open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
            status.update(status='running',task=f'{name}-{task}',child_pid=child.pid,task_started_at=stamp());save()
            while child.poll() is None:time.sleep(5);save()
            status['completed'].append({'task':task,'exit_code':child.returncode,'finished_at':stamp()});save()
            if child.returncode:status['status']='failed';save();return
    evaluation=json.loads((OUT/'evaluation'/f'validation-{name}'/'status.json').read_text())
    if evaluation['status']!='completed':status.update(status='incomplete_validation');save();return
    child=subprocess.Popen([py,str(ROOT/'scripts/goal_v2_analyze_validation.py')],stdout=(OUT/f'{name}-analysis.log').open('x'),stderr=subprocess.STDOUT,env=env,cwd=ROOT)
    status.update(task=f'{name}-analysis',child_pid=child.pid);save()
    while child.poll() is None:time.sleep(5);save()
    status['status']='completed' if child.returncode==0 else 'failed';save()
if __name__=='__main__':main()

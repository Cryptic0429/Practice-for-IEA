"""Wait on the verified feature process, then execute bounded CPU numeric-head pilots."""
import datetime,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def main():
    out=OUT/'worker-numerical.json';assert not out.exists();state={'pid':os.getpid(),'status':'waiting_features','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    def write():
        state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();t=out.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(out)
    write();start=time.monotonic()
    while True:
        status=OUT/'frozen-A-features-v1/status.json'
        if status.exists():
            current=json.loads(status.read_text())
            if current['status']=='completed':break
            if current['status']=='failed' or not Path(f'/proc/{current["pid"]}').exists():state.update(status='feature_extraction_failed',feature_state=current);write();return
        if time.monotonic()-start>4000:state['status']='feature_observation_limit';write();return
        time.sleep(5)
    py=str(ROOT/'envs/iea-rl/bin/python');env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONPATH=str(ROOT/'scripts'))
    jobs=[('train64',['goal_v2_train_numerical_head.py','--scope','64']),
          ('probe64',['goal_v2_evaluate_numerical_head.py','--name','numeric-head64-s20261103','--split','train_probe']),
          ('train512',['goal_v2_train_numerical_head.py','--scope','512']),
          ('validate512',['goal_v2_evaluate_numerical_head.py','--name','numeric-head512-s20261103','--split','validation'])]
    for name,args in jobs:
        with (OUT/f'numerical-{name}.log').open('x') as log:
            child=subprocess.Popen([py,str(ROOT/'scripts'/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
            state.update(status='running',task=name,child_pid=child.pid);write()
            while child.poll() is None:time.sleep(5);write()
            if child.returncode:state.update(status='failed',exit_code=child.returncode);write();return
    state['status']='completed';write()
if __name__=='__main__':main()

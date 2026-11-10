"""Evaluate the two predeclared final checkpoints once; no epoch search."""
import os
import subprocess
import time
from functional_v1.common import ROOT,read,write,stamp
from .runtime import EXP

def main():
    receipt=EXP/'final-queue.json';assert not receipt.exists()
    state=dict(at=stamp(),pid=os.getpid(),status='waiting_for_final_checkpoints',jobs=[]);write(receipt,state)
    waiting=['language','language_merger'];deadline=time.monotonic()+3600
    while waiting:
        assert time.monotonic()<deadline,'Owned final queue timed out'
        ready=[]
        for route in waiting:
            run=EXP/'runs'/f'{route}-seed-20261121'
            if not (run/'status.json').exists():continue
            status=read(run/'status.json');assert status['status'] not in ('failed','budget_stopped'),status
            if status['status']=='completed':
                assert status['step']==240 and (run/'checkpoint-240/adapter_model.safetensors').exists()
                ready.append(route)
        if not ready:time.sleep(5);continue
        route=ready[0];model=route.replace('_','-')+'-e2'
        command=[str(ROOT/'envs/iea-rl/bin/python'),'-m','functional_v10.evaluate','--route',route,'--adapter',
            f'experiments/functional-v10-20261106/runs/{route}-seed-20261121/checkpoint-240','--model',model,'--gpu','2']
        with (EXP/f'evaluate-{model}.log').open('wb') as output:
            child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT/'scripts'),CUDA_VISIBLE_DEVICES='2'),stdout=output,stderr=subprocess.STDOUT)
            state.update(status='evaluating',current=model);state['jobs'].append(dict(model=model,pid=child.pid,command=command,started_at=stamp()));write(receipt,state)
            code=child.wait()
        state['jobs'][-1].update(returncode=code,ended_at=stamp());write(receipt,state)
        assert code==0,'Evaluation failed; preserve and stop queue'
        waiting.remove(route)
    state.update(status='completed',finished_at=stamp(),current=None);write(receipt,state)

if __name__=='__main__':main()

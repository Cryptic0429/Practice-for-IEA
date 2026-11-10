"""Evaluate the single predeclared final checkpoint once; no epoch search."""
import os
import subprocess
import time
from functional_v1.common import ROOT,read,write,stamp
from .runtime import EXP

def main():
    receipt=EXP/'final-queue.json';assert not receipt.exists()
    state=dict(at=stamp(),pid=os.getpid(),status='waiting_for_final_checkpoints',jobs=[]);write(receipt,state)
    waiting=['language_merger'];deadline=time.monotonic()+3600
    while waiting:
        assert time.monotonic()<deadline,'Owned final queue timed out'
        ready=[]
        for route in waiting:
            run=EXP/'runs'/f'{route}-seed-20261125'
            if not (run/'status.json').exists():continue
            status=read(run/'status.json');assert status['status'] not in ('failed','budget_stopped'),status
            if status['status']=='completed':
                assert status['step']==125 and (run/'checkpoint-125/adapter_model.safetensors').exists()
                used=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                if used<512:ready.append(route)
        if not ready:time.sleep(5);continue
        route=ready[0];model=route.replace('_','-')+'-e1'
        command=[str(ROOT/'envs/iea-rl/bin/python'),'-m','functional_v12.evaluate','--route',route,'--adapter',
            f'experiments/functional-v12-20261106/runs/{route}-seed-20261125/checkpoint-125','--model',model,'--gpu','0']
        with (EXP/f'evaluate-{model}.log').open('wb') as output:
            child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT/'scripts'),CUDA_VISIBLE_DEVICES='0'),stdout=output,stderr=subprocess.STDOUT)
            state.update(status='evaluating',current=model);state['jobs'].append(dict(model=model,pid=child.pid,command=command,started_at=stamp()));write(receipt,state)
            code=child.wait()
        state['jobs'][-1].update(returncode=code,ended_at=stamp());write(receipt,state)
        assert code==0,'Evaluation failed; preserve and stop queue'
        waiting.remove(route)
    state.update(status='completed',finished_at=stamp(),current=None);write(receipt,state)

if __name__=='__main__':main()

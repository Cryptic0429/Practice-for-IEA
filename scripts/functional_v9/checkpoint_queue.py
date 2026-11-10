"""Evaluate all prespecified epochs as they finish; no metric-based filtering."""
import os
from pathlib import Path
import subprocess
import time
from functional_v1.common import ROOT, read, write, stamp
from .runtime import EXP


def main():
    receipt=EXP/'checkpoint-queue.json'; assert not receipt.exists()
    state=dict(at=stamp(),pid=os.getpid(),status='waiting_for_prespecified_checkpoints',jobs=[])
    write(receipt,state)
    waiting={(route,epoch) for route in ('base','author') for epoch in range(1,5)}
    deadline=time.monotonic()+7200
    while waiting:
        assert time.monotonic()<deadline, 'Owned queue timed out; no endless retries'
        baseline=read(EXP/'baseline-queue.json')
        if baseline['status']!='completed':
            assert all(j.get('returncode',0)==0 for j in baseline['jobs'])
            time.sleep(5);continue
        ready=[]
        for route,epoch in sorted(waiting,key=lambda pair:(pair[1],pair[0])):
            folder=EXP/'runs'/f'{route}-seed-20261115'
            status=read(folder/'status.json')
            assert status['status'] not in ('failed','budget_stopped'), status
            step=231*epoch
            adapter=folder/f'checkpoint-{step}'
            if (status.get('step',0)>step or status['status']=='completed') and (adapter/'adapter_model.safetensors').exists() and (adapter/'adapter_config.json').exists():
                ready.append((route,epoch,adapter))
        if not ready:
            time.sleep(5);continue
        route,epoch,adapter=ready[0]
        model=f'{route}-e{epoch}'
        command=[str(ROOT/'envs/iea-rl/bin/python'),'-m','functional_v9.evaluate','--origin',route,
                 '--adapter',str(adapter.relative_to(ROOT)),'--model',model,'--gpu','2']
        with (EXP/f'evaluate-{model}.log').open('wb') as output:
            child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT/'scripts'),CUDA_VISIBLE_DEVICES='2'),stdout=output,stderr=subprocess.STDOUT)
            state.update(status='evaluating',current=model)
            state['jobs'].append(dict(model=model,pid=child.pid,command=command,started_at=stamp()))
            write(receipt,state)
            code=child.wait()
        state['jobs'][-1].update(returncode=code,ended_at=stamp());write(receipt,state)
        assert code==0, 'Evaluation failed; preserve error and stop queue'
        waiting.remove((route,epoch))
    state.update(status='completed',finished_at=stamp(),current=None);write(receipt,state)

if __name__=='__main__':
    main()

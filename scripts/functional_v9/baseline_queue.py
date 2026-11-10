"""Sequential matched-origin baselines on reserved experiment GPU2."""
import os
import subprocess
from functional_v1.common import ROOT, write, stamp
from .runtime import EXP


def main():
    receipt = EXP/'baseline-queue.json'
    assert not receipt.exists()
    state = dict(at=stamp(),pid=os.getpid(),status='running',jobs=[])
    write(receipt,state)
    for origin in ('base','author'):
        command=[str(ROOT/'envs/iea-rl/bin/python'),'-m','functional_v9.evaluate','--origin',origin,
                 '--model',origin+'-init','--gpu','2']
        with (EXP/f'evaluate-{origin}-init.log').open('wb') as output:
            p=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT/'scripts'),CUDA_VISIBLE_DEVICES='2'),stdout=output,stderr=subprocess.STDOUT)
            state['jobs'].append(dict(origin=origin,pid=p.pid,command=command,started_at=stamp()))
            write(receipt,state)
            code=p.wait()
        state['jobs'][-1].update(returncode=code,ended_at=stamp())
        write(receipt,state)
        assert code==0, 'Baseline failed; no silent retry'
    state.update(status='completed',finished_at=stamp());write(receipt,state)

if __name__=='__main__':
    main()

"""Bounded development prediction queue, never reads final acceptance outcomes."""
import os
from pathlib import Path
import subprocess
import time
from .common import ROOT, EXP, EVAL, read, write, stamp


def main():
    queue=[(start,epoch) for epoch in range(1,5) for start in ('base','current')]
    began=time.monotonic()
    log_dir=EXP/'development-queue'
    log_dir.mkdir(exist_ok=True)
    while queue:
        if (EXP/'BUDGET_STOP').exists() or time.monotonic()-began>10800:
            write(log_dir/'status.json',{'status':'stopped','pending':queue,'at':stamp()});return
        ready=[]
        for start,epoch in queue:
            checkpoint=EXP/'runs'/f'{start}-seed-20261104'/f'checkpoint-{epoch*110}'
            if (checkpoint/'trainer_state.json').exists() and (checkpoint/'adapter_model.safetensors').exists():
                ready.append((start,epoch,checkpoint))
        if not ready:
            time.sleep(5);continue
        start,epoch,checkpoint=ready[0]
        model_id=f'{start}-e{epoch}'
        env=dict(os.environ,PYTHONPATH=str(ROOT/'scripts'),CUDA_VISIBLE_DEVICES='3',TOKENIZERS_PARALLELISM='false')
        with (log_dir/f'{model_id}.log').open('a') as log:
            process=subprocess.run([str(ROOT/'envs/iea-rl/bin/python'),'-u','-m','functional_v1.evaluate','predict',
                '--model',model_id,'--adapter',str(checkpoint),'--split','development','--gpu','3'],env=env,stdout=log,stderr=subprocess.STDOUT)
        if process.returncode:
            write(log_dir/'status.json',{'status':'failed','model_id':model_id,'pending':queue,'at':stamp()});return
        queue.remove((start,epoch))
        write(log_dir/'status.json',{'status':'running' if queue else 'completed','last':model_id,'pending':queue,'at':stamp()})


if __name__=='__main__':main()

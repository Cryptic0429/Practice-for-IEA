"""Evaluate saved epochs in fresh processes; waiting never retains a CUDA model."""
import os,subprocess,sys,time
from .common import ROOT,EXP,EVAL,read,write,stamp

def main():
    pending=[(route,epoch) for epoch in range(1,5) for route in ('curriculum','control')]
    deadline=time.monotonic()+10800
    while pending:
        if time.monotonic()>deadline:raise TimeoutError('Development queue deadline; keep partial outputs')
        progressed=False
        for route,epoch in list(pending):
            run=EXP/'runs'/f'{route}-seed-20261106'
            checkpoint=run/f'checkpoint-{180*epoch}'
            weights=checkpoint/'adapter_model.safetensors'
            if not weights.exists() or time.time()-weights.stat().st_mtime<3:continue
            model=f'{route}-e{epoch}'
            write(EXP/'development-queue.json',{'status':'predicting','model':model,'at':stamp(),'remaining':pending})
            with (EXP/f'eval-{model}.log').open('w') as log:
                result=subprocess.run([sys.executable,'-m','functional_v2.evaluate','--model',model,'--adapter',str(checkpoint),'--gpu','3'],
                   cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'3'},stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('Failed development job: '+model)
            pending.remove((route,epoch));progressed=True
        for route,epoch in pending:
            status=EXP/'runs'/f'{route}-seed-20261106'/'status.json'
            if status.exists() and read(status)['status'] in ('failed','budget_stopped'):
                raise RuntimeError('Training stopped; preserve development queue partial state')
        if not progressed:time.sleep(10)
    write(EXP/'development-queue.json',{'status':'completed','at':stamp(),'models':[f'{r}-e{e}' for e in range(1,5) for r in ('curriculum','control')]})
if __name__=='__main__':main()

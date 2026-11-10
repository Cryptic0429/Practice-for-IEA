"""CPU queue after completed cache. Stop on any failed command, no adaptive fitting."""
import os, subprocess, sys, time
from .common import ROOT, EXP, read, write, stamp

def main():
    while not (EXP/'features/status.json').exists() or read(EXP/'features/status.json')['status']!='completed':
        time.sleep(5)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    jobs=[]
    for route in ('last','request_mean'):
        jobs.append(['functional_v20.train','--route',route])
        for epoch in (20,80):
            jobs.append(['functional_v20.evaluate','--model',f'{route.replace("_","-")}-e{epoch}',
                         '--checkpoint',f'experiments/functional-v20-20261107/runs/{route}/checkpoint-{epoch}.pt'])
    jobs += [['functional_v20.seed_reviews'], ['functional_v20.visual_review','prepare','--split','old-native',
             '--models','last-e20','last-e80','request-mean-e20','request-mean-e80','--name','joint-routes-epochs']]
    results=[]
    for args in jobs:
        write(EXP/'cpu-queue.json',dict(status='running',current=args,jobs=results,at=stamp(),GPU_calls=0))
        result=subprocess.run([sys.executable,'-u','-m',*args],cwd=ROOT)
        results.append(dict(args=args,returncode=result.returncode))
        write(EXP/'cpu-queue.json',dict(status='running' if result.returncode==0 else 'failed',jobs=results,at=stamp(),GPU_calls=0))
        if result.returncode:raise SystemExit(result.returncode)
    write(EXP/'cpu-queue.json',dict(status='completed',jobs=results,at=stamp(),GPU_calls=0))

if __name__=='__main__':main()

"""One fixed CPU training, four fixed policy/checkpoint old-request evaluations."""
import os,subprocess,sys
from .common import ROOT,EXP,read,write,stamp

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and (EXP/'texture-features.json').is_file()
    jobs=[['functional_v21.train']]
    for epoch in (20,80):
        for policy in ('magnitude','magnitude-quality'):
            jobs.append(['functional_v21.evaluate','--model',f'{policy}-e{epoch}','--policy',policy,
                '--head',f'experiments/functional-v21-20261107/runs/seed-20261205/checkpoint-{epoch}.pt'])
    jobs += [['functional_v21.seed_reviews'],['functional_v21.visual_review','prepare','--split','old-native',
        '--models','magnitude-e20','magnitude-quality-e20','magnitude-e80','magnitude-quality-e80','--name','shared-head-policies']]
    results=[]
    for args in jobs:
        write(EXP/'cpu-queue.json',dict(at=stamp(),status='running',current=args,jobs=results,GPU_calls=0))
        result=subprocess.run([sys.executable,'-u','-m',*args],cwd=ROOT)
        results.append(dict(args=args,returncode=result.returncode))
        write(EXP/'cpu-queue.json',dict(at=stamp(),status='running' if result.returncode==0 else 'failed',jobs=results,GPU_calls=0))
        if result.returncode:raise SystemExit(result.returncode)
    write(EXP/'cpu-queue.json',dict(at=stamp(),status='completed',jobs=results,GPU_calls=0))

if __name__=='__main__':main()

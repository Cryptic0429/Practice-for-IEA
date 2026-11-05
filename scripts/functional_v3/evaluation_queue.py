"""Bounded per-route eval queue; each job exits to release all GPU allocations."""
import argparse,json,os,subprocess,time,traceback
from .common import ROOT,EXP,read,write,stamp


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--route',choices=['dpo','sft'],required=True);ap.add_argument('--gpu',choices=['0','2','3'],required=True);a=ap.parse_args()
    status={'status':'waiting_for_training','route':a.route,'gpu':a.gpu,'at':stamp()};file=EXP/f'evaluation-queue-{a.route}.json';write(file,status)
    run=EXP/'runs'/f'{a.route}-seed-20261107';deadline=time.monotonic()+5400
    try:
        while time.monotonic()<deadline:
            s=read(run/'status.json') if (run/'status.json').exists() else {}
            if s.get('status') in ('failed','budget_stopped'):raise RuntimeError('Training did not complete: '+str(s.get('error',s['status'])))
            if s.get('status')=='completed':
                used=int(subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                if used<512:break
            time.sleep(10)
        else:raise TimeoutError('Training/evaluation queue deadline reached')
        checkpoints=sorted(run.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]));assert len(checkpoints)==2
        for epoch,adapter in enumerate(checkpoints,1):
            for split in ('development','natural-use'):
                mid=f'{a.route}-e{epoch}';status.update(status='predicting',model_id=mid,split=split,at=stamp());write(file,status)
                cmd=[str(ROOT/'envs/iea-rl/bin/python'),'-u','-m','functional_v3.evaluate','--model',mid,'--adapter',str(adapter),'--gpu',a.gpu,'--split',split]
                with (EXP/f'eval-{split}-{mid}.log').open('w') as log:
                    subprocess.run(cmd,cwd=ROOT,env={**os.environ,'PYTHONPATH':'scripts','CUDA_VISIBLE_DEVICES':a.gpu},stdout=log,stderr=log,check=True,timeout=1800)
        status.update(status='completed',at=stamp());write(file,status)
    except BaseException as error:
        status.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),at=stamp());write(file,status);raise
if __name__=='__main__':main()

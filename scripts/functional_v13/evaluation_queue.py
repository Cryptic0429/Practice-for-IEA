"""Wait for complete epoch snapshots; evaluate each predeclared checkpoint once."""
import os
import subprocess
import time
from functional_v1.common import ROOT,read,write,sha,stamp
from .runtime import EXP,ROUTES


def main():
    receipt=EXP/'evaluation-queue.json';assert not receipt.exists()
    protocol=read(EXP/'training-protocol.json');s=protocol['training']
    pending=[(r,e) for e in (1,2) for r in ROUTES]
    state=dict(at=stamp(),pid=os.getpid(),status='waiting_for_epoch_snapshots',jobs=[]);write(receipt,state)
    deadline=time.monotonic()+7200
    while pending:
        assert time.monotonic()<deadline,'Queue deadline reached; preserve status'
        ready=[]
        for route,epoch in pending:
            run=EXP/'runs'/f'{route}-seed-{s["seed"]}'
            checkpoint=run/f'checkpoint-{s["steps_per_epoch"]*epoch}'
            if (run/'status.json').exists():
                status=read(run/'status.json')
                assert status['status'] not in ('failed','budget_stopped'),status
            # Trainer saves trainer_state last; never load a half-written adapter.
            if (checkpoint/'trainer_state.json').exists():
                saved=read(checkpoint/'trainer_state.json')
                assert saved['global_step']==s['steps_per_epoch']*epoch
                assert (checkpoint/'adapter_config.json').exists()
                digest=sha(checkpoint/'adapter_model.safetensors')
                ready.append((route,epoch,checkpoint,digest))
        if not ready:time.sleep(5);continue
        used=int(subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if used>=512:time.sleep(5);continue
        route,epoch,checkpoint,digest=ready[0];mid=route.replace('_','-')+f'-e{epoch}'
        command=[str(ROOT/'envs/iea-rl/bin/python'),'-m','functional_v13.evaluate','--route',route,
            '--adapter',str(checkpoint.relative_to(ROOT)),'--model',mid,'--gpu','2']
        with (EXP/f'evaluate-{mid}.log').open('wb') as output:
            child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT/'scripts'),CUDA_VISIBLE_DEVICES='2'),stdout=output,stderr=subprocess.STDOUT)
            state.update(status='evaluating',current=mid)
            state['jobs'].append(dict(model=mid,epoch=epoch,route=route,adapter_sha256=digest,pid=child.pid,command=command,started_at=stamp()))
            write(receipt,state);code=child.wait()
        state['jobs'][-1].update(returncode=code,ended_at=stamp());write(receipt,state)
        assert code==0,'Evaluation failed; stop queue and preserve all predictions'
        assert sha(checkpoint/'adapter_model.safetensors')==digest
        pending.remove((route,epoch))
    state.update(status='completed',finished_at=stamp(),current=None);write(receipt,state)


if __name__=='__main__':main()

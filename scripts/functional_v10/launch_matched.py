"""Start the two authorized matched routes; GPU1 is never selected."""
import os
import subprocess
from functional_v1.common import ROOT, write, stamp
from .runtime import EXP


def main():
    assert not (EXP / 'launch.json').exists()
    from functional_v1.common import read,GPUJob,EXP as BUDGET_EXP
    assert 16-GPUJob.consumed(read(BUDGET_EXP/'resources.json'))>2.0
    assert read(EXP/'training-protocol.json')['training']['steps']==240
    processes = []
    for route, gpu in [('language','0'),('language_merger','3')]:
        log = EXP / f'train-{route}.log'
        env = dict(os.environ, PYTHONPATH=str(ROOT/'scripts'), CUDA_VISIBLE_DEVICES=gpu, TOKENIZERS_PARALLELISM='false')
        args = [str(ROOT/'envs/iea-rl/bin/python'),'-m','functional_v10.train','--route',route,'--gpu',gpu]
        with log.open('wb') as output:
            p = subprocess.Popen(args,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(dict(route=route,gpu=gpu,pid=p.pid,log=str(log.relative_to(ROOT)),command=args))
    write(EXP/'launch.json',dict(at=stamp(),processes=processes,original_GPU1_untouched=True))
    print(processes)

if __name__ == '__main__':
    main()

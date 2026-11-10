"""Restart only recorded, idle experiment services; preserve original GPU1 service."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import requests
from functional_v1.common import ROOT,read,write,sha,stamp

EXP=ROOT/'experiments/functional-v10-20261106'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--previous',type=Path,required=True)
    args=parser.parse_args()
    assert args.name.replace('-','').isalnum()
    receipt=EXP/'demo'/f'{args.name}-service-restart.json';assert not receipt.exists()
    previous=read(ROOT/args.previous)
    health=requests.get('http://127.0.0.1:18282/health',timeout=10).json()
    assert health['loaded_model'] is None, 'Never interrupt an active trial model'
    old=previous['new']
    assert {r['module'] for r in old}=={'functional_v1.worker','functional_v1.web'}
    for record in old:
        pid=record['pid'];proc=Path('/proc')/str(pid)
        assert proc.resolve().exists() and (proc/'cwd').resolve()==ROOT.resolve()
        assert record['module'].encode() in (proc/'cmdline').read_bytes().split(b'\0')
        assert pid!=3378799
    for record in old:os.kill(record['pid'],signal.SIGTERM)
    time.sleep(2)
    new=[]
    for module,gpu,python,extra in [('functional_v1.worker','2','iea-rl',['--gpu','2','--port','18282']),('functional_v1.web','','iea',['--port','12346'])]:
        env=dict(os.environ,PYTHONPATH=str(ROOT/'scripts'),CUDA_VISIBLE_DEVICES=gpu,TOKENIZERS_PARALLELISM='false')
        log=EXP/'demo'/f'{args.name}-{module.rsplit(".",1)[1]}.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('wb') as output:
            p=subprocess.Popen([str(ROOT/f'envs/{python}/bin/python'),'-u','-m',module,*extra],cwd=ROOT,env=env,
                stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        new.append(dict(module=module,pid=p.pid))
    write(EXP/'demo'/f'{args.name}-service-launch.json',dict(at=stamp(),new=new,old_owned_pids=[r['pid'] for r in old]))
    for _ in range(60):
        try:
            health=requests.get('http://127.0.0.1:18282/health',timeout=2).json()
            catalog=requests.get('http://127.0.0.1:12346/api/models',timeout=2).json()
            assert requests.get('http://127.0.0.1:12346/v10',timeout=5).status_code==200
            break
        except (requests.RequestException,AssertionError):time.sleep(1)
    else:raise RuntimeError('Owned services did not become ready; inspect logs')
    assert catalog['default']=='functional-current-20261104'
    author=Path('/proc/3378799')
    gpu1_pids=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
    assert author.exists() and '3378799' in gpu1_pids
    write(receipt,dict(at=stamp(),old_owned_pids=[r['pid'] for r in old],new=new,worker_health=health,
        registered_models=len(catalog['models']),default_unchanged=catalog['default'],
        original_author_service_pid_preserved=3378799,previous_receipt=str(args.previous),previous_receipt_sha256=sha(ROOT/args.previous)))
    print(read(receipt))

if __name__=='__main__':main()

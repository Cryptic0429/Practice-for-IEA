"""Restart only identified idle experimental services; preserve GPU1 author's service."""
import os
import signal
import subprocess
import time
from pathlib import Path
import requests
from functional_v1.common import ROOT,read,write,stamp,sha
from .runtime import EXP


def main():
    receipt=EXP/'demo/service-restart.json';assert not receipt.exists()
    previous=ROOT/'experiments/functional-v14-20261106/demo/service-restart.json'
    p=read(previous)
    health=requests.get('http://127.0.0.1:18282/health',timeout=10).json()
    assert health['loaded_model'] is None,'Do not interrupt an active model; wait for idle unload'
    old=p['new']
    assert {x['module'] for x in old}=={'functional_v1.worker','functional_v1.web'}
    for x in old:
        proc=Path('/proc')/str(x['pid']);assert proc.exists() and (proc/'cwd').resolve()==ROOT.resolve()
        assert x['module'].encode() in (proc/'cmdline').read_bytes().split(b'\0') and x['pid']!=3378799
    for x in old:os.kill(x['pid'],signal.SIGTERM)
    time.sleep(2);new=[]
    for module,gpu,python,args in [('functional_v1.worker','2','iea-rl',['--gpu','2','--port','18282']),('functional_v1.web','','iea',['--port','12346'])]:
        log=EXP/'demo'/f'{module.rsplit(".",1)[1]}.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('wb') as output:
            child=subprocess.Popen([str(ROOT/f'envs/{python}/bin/python'),'-u','-m',module,*args],cwd=ROOT,
                env=dict(os.environ,PYTHONPATH=str(ROOT/'scripts'),CUDA_VISIBLE_DEVICES=gpu,TOKENIZERS_PARALLELISM='false'),
                stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        new.append(dict(module=module,pid=child.pid))
    write(EXP/'demo/service-launch.json',dict(at=stamp(),new=new,old=old))
    for _ in range(50):
        try:
            h=requests.get('http://127.0.0.1:18282/health',timeout=2).json()
            c=requests.get('http://127.0.0.1:12346/api/models',timeout=2).json()
            assert len(c['models'])==25 and c['default']=='functional-current-20261104'
            assert requests.get('http://127.0.0.1:12346/v15',timeout=5).status_code==200
            break
        except (requests.RequestException,AssertionError):time.sleep(1)
    else:raise RuntimeError('Experimental services not ready; inspect recorded owned-service logs')
    assert Path('/proc/3378799').exists()
    pids=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
    assert '3378799' in pids
    write(receipt,dict(at=stamp(),new=new,old=old,previous_receipt=str(previous.relative_to(ROOT)),previous_sha256=sha(previous),
        worker_health=h,models_registered=25,default_unchanged=c['default'],original_GPU1_PID_preserved=3378799))
    print(read(receipt))


if __name__=='__main__':main()

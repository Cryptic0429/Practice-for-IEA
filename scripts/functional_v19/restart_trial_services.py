"""Load trial code only after a reviewed R19 adapter is concretely registered."""
import argparse,os,signal,subprocess,time
from pathlib import Path
import requests
from functional_v1.common import ROOT,read,write,stamp
from functional_v1.catalog import models
EXP=ROOT/'experiments/functional-v19-20261107'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--old-worker-pid',required=True,type=int);parser.add_argument('--old-web-pid',required=True,type=int)
    args=parser.parse_args();target=EXP/'demo/service-restart.json';assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True)
    selection=read(EXP/'development-selection.json');assert selection['models']
    for meta in selection['models'].values():assert (ROOT/meta['adapter']/'adapter_model.safetensors').exists()
    count=len(models())
    assert requests.get('http://127.0.0.1:18282/health',timeout=10).json()['loaded_model'] is None
    old=[dict(module='functional_v1.worker',pid=args.old_worker_pid,required='--gpu 2 --port 18282'),
         dict(module='functional_v1.web',pid=args.old_web_pid,required='--port 12346')]
    for item in old:
        assert item['pid']!=3378799
        proc=Path('/proc')/str(item['pid']);assert (proc/'cwd').resolve()==ROOT.resolve()
        command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        assert item['module'] in command and item['required'] in command
    # Receipts survive interruption before launch. Original author service is
    # outside both validated experimental process identities.
    write(EXP/'demo/service-restart-preflight.json',dict(at=stamp(),old=old,expected_catalog_count=count,idle_worker=True))
    for item in old:os.kill(item['pid'],signal.SIGTERM)
    time.sleep(1);new=[]
    for module,environment,gpu,arguments in [('functional_v1.worker','iea-rl','2',['--gpu','2','--port','18282']),('functional_v1.web','iea','',['--port','12346'])]:
        with (EXP/'demo'/f'{module.rsplit(".",1)[1]}.log').open('ab') as output:
            child=subprocess.Popen([str(ROOT/f'envs/{environment}/bin/python'),'-u','-m',module,*arguments],cwd=ROOT,
                env=dict(os.environ,PYTHONPATH=str(ROOT/'scripts'),CUDA_VISIBLE_DEVICES=gpu),stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        new.append(dict(module=module,pid=child.pid))
    for _ in range(40):
        try:
            health=requests.get('http://127.0.0.1:18282/health',timeout=1).json()
            catalog=requests.get('http://127.0.0.1:12346/api/models',timeout=1).json()
            assert len(catalog['models'])==count and catalog['default']=='functional-current-20261104'
            assert set(selection['models'])<={m['id'] for m in catalog['models']}
            assert requests.get('http://127.0.0.1:12346/v19',timeout=3).status_code==200
            break
        except (requests.RequestException,AssertionError):time.sleep(.2)
    else:raise RuntimeError('Replacement service not ready; inspect owned-service logs')
    assert '3378799' in subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
    write(target,dict(at=stamp(),old=old,new=new,worker=health,models_registered=count,default_unchanged=True,original_GPU1_PID_preserved=3378799))
    print(new)

if __name__=='__main__':main()

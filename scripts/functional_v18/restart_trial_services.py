"""Replace only the idle experimental CPU services after adding R18 provenance."""
import os,signal,subprocess,time
from pathlib import Path
import requests
from functional_v1.common import ROOT,write,stamp
from .runtime import EXP

def main():
    target=EXP/'demo/service-restart.json';assert not target.exists()
    target.parent.mkdir(parents=True,exist_ok=True)
    live=[]
    for pid in (4111921,4121909):
        proc=Path('/proc')/str(pid)
        if proc.exists() and (proc/'cmdline').read_bytes():live.append(pid)
    if 4111921 in live:
        assert requests.get('http://127.0.0.1:18282/health',timeout=10).json()['loaded_model'] is None
    old=[dict(module='functional_v1.worker',pid=4111921),dict(module='functional_v1.web',pid=4121909)]
    for item in old:
        if item['pid'] not in live:continue
        proc=Path('/proc')/str(item['pid']);assert (proc/'cwd').resolve()==ROOT.resolve()
        assert item['module'].encode() in (proc/'cmdline').read_bytes().split(b'\0')
    for item in old:
        if item['pid'] in live:os.kill(item['pid'],signal.SIGTERM)
    time.sleep(1)
    new=[]
    for module,python,gpu,args in [('functional_v1.worker','iea-rl','2',['--gpu','2','--port','18282']),('functional_v1.web','iea','',['--port','12346'])]:
        with (EXP/'demo'/f'{module.rsplit(".",1)[1]}.log').open('ab') as output:
            p=subprocess.Popen([str(ROOT/f'envs/{python}/bin/python'),'-u','-m',module,*args],cwd=ROOT,
                env=dict(os.environ,PYTHONPATH=str(ROOT/'scripts'),CUDA_VISIBLE_DEVICES=gpu),stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        new.append(dict(module=module,pid=p.pid))
    for _ in range(40):
        try:
            health=requests.get('http://127.0.0.1:18282/health',timeout=1).json()
            catalog=requests.get('http://127.0.0.1:12346/api/models',timeout=1).json()
            assert len(catalog['models'])==27 and catalog['default']=='functional-current-20261104'
            assert requests.get('http://127.0.0.1:12346/v18',timeout=3).status_code==200
            break
        except (requests.RequestException,AssertionError):time.sleep(.2)
    else:raise RuntimeError('Experimental service restart failed; see log')
    assert Path('/proc/3378799').exists()
    assert '3378799' in subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
    write(target,dict(at=stamp(),old=old,new=new,worker=health,models_registered=27,default_unchanged=True,original_GPU1_PID_preserved=3378799))
    print(new)

if __name__=='__main__':main()

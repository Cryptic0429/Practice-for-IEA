"""Restart only the authorized experimental Flask process, preserving workers."""
import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from functional_v1.common import ROOT,read,write,stamp

EXP=ROOT/'experiments/functional-v18-20261107'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--old-web-pid',type=int,required=True)
    parser.add_argument('--receipt-name',default='progress-page-service');args=parser.parse_args()
    assert args.receipt_name.replace('-','').isalnum()
    target=EXP/(args.receipt_name+'.json');assert not target.exists()
    pid=args.old_web_pid
    cmd=subprocess.check_output(['ps','-p',str(pid),'-o','args='],text=True).strip()
    assert 'functional_v1.web --port 12346' in cmd and 'envs/iea/bin/python' in cmd
    assert pid not in (3378799,4111921,4119157,4119696,4120161,3879191)
    os.kill(pid,signal.SIGTERM)
    for _ in range(30):
        old=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True)
        if old.returncode!=0 or old.stdout.strip().startswith('Z'):break
        time.sleep(.1)
    else:raise RuntimeError('Old experimental web did not terminate; no replacement launched')
    env=dict(os.environ,PYTHONPATH=str(ROOT/'scripts'),CUDA_VISIBLE_DEVICES='')
    log=(EXP/'web-progress.log').open('ab')
    child=subprocess.Popen([str(ROOT/'envs/iea/bin/python'),'-u','-m','functional_v1.web','--port','12346'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    result=None
    for _ in range(50):
        assert child.poll() is None,'Replacement web exited; inspect log'
        try:
            with urllib.request.urlopen('http://127.0.0.1:12346/api/v18/experiment',timeout=1) as response:
                result=json.load(response)
            break
        except OSError:time.sleep(.1)
    assert result and result['admission']['unique_semantic_inputs']==925
    assert result['admission']['preference_semantics']==19
    write(target,dict(at=stamp(),status='experimental_records_page_live',old_web_pid=pid,new_web_pid=child.pid,
        worker_untouched=4111921,original_GPU1_service_untouched=3378799,training_PIDs_untouched=[4119157,4119696],
        port=12346,binding='127.0.0.1',url='http://127.0.0.1:12346/v18',
        model_catalog_unchanged=True,new_untrained_models_registered=False,model_catalog_readiness='Two completed failed R18 models were registered separately before this web restart; no default promotion',source='Actual local HTTP925source records/19preferences confirmed',
        GPU_calls=0,browser_validation_completed=False))
    print('Experimental Flask records live,PID',child.pid,';worker/original demo/trainers untouched.')


if __name__=='__main__':main()

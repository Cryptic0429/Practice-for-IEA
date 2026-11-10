"""Refresh only the loopback trial web process; keep worker idle and old demo alive."""
import os,signal,time,subprocess,io
from pathlib import Path
import requests
from .common import *
from functional_v1.common import EXP as BUDGET

def main():
 old=4177352;proc=Path('/proc')/str(old);assert old!=3378799 and (proc/'cwd').resolve()==ROOT.resolve();cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode();assert 'functional_v1.web --port 12346' in cmd
 assert (BUDGET/'BUDGET_STOP.json').exists() and requests.get('http://127.0.0.1:18282/health',timeout=10).json()['loaded_model'] is None
 os.kill(old,signal.SIGTERM);time.sleep(.4)
 with (EXP/'demo/final-web.log').open('ab') as log:
  child=subprocess.Popen([str(ROOT/'envs/iea/bin/python'),'-u','-m','functional_v1.web','--port','12346'],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT/'scripts'),CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 url='http://127.0.0.1:12346'
 for _ in range(30):
  try:r=requests.get(url+'/api/models',timeout=1);r.raise_for_status();catalog=r.json();break
  except requests.RequestException:time.sleep(.2)
 else:raise RuntimeError('Trial web process did not start')
 assert catalog['default']=='functional-current-20261104' and not catalog['inference_available'] and len(catalog['models'])==37
 assert requests.get(url+'/v23?kind=evaluation',timeout=10).status_code==200
 assert requests.get(url+'/v22?kind=evaluation',timeout=10).status_code==200
 denied=requests.post(url+'/api/edit',files={'image':('budget-test.png',b'budget-test','image/png')},data={'instruction':'keep','model_id':'functional-v23-quality-20261208'},timeout=10)
 assert denied.status_code==503 and denied.json()['code']=='budget_exhausted' and not denied.json().get('request_id') and denied.json()['after'] is None
 assert requests.get(url+'/files/d95cfb5a88b646b1b7bee4f1cb764ae4/output.png',timeout=10).status_code==200
 assert '3378799' in subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
 write(EXP/'demo/final-service-status.json',dict(at=stamp(),old_trial_web_pid=old,new_trial_web_pid=child.pid,catalog_count=len(catalog['models']),default=catalog['default'],new_inference=False,budget_test_http_status=denied.status_code,budget_test_no_request_created=True,saved_results_available=True,original_service_PID=3378799,original_service_preserved=True))
 print('Live saved-result pages and downloads work; new edits denied by budget before creating request; GPU1 preserved.')
if __name__=='__main__':main()

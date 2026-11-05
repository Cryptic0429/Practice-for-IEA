"""One bounded read of actual remote processes and numerical state, no mutations remotely."""
import datetime,json,subprocess
from pathlib import Path
REMOTE=r'''
import datetime,json,math,os
from pathlib import Path
root=Path('/mnt/sdc/zhc/IEA/outputs/goal-v2-20261103')
result={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'workers':{},'runs':{},'calibration':{},'auxiliary':{}}
for p in sorted(root.glob('worker-*.json'))+([root/'cpu-queue.json'] if (root/'cpu-queue.json').exists() else []):
 d=json.loads(p.read_text())
 for key in ['pid','child_pid']:
  if key in d:
   proc=Path('/proc')/str(d[key]);d[key+'_live']=proc.exists()
   if proc.exists():d[key+'_command']=proc.joinpath('cmdline').read_bytes().decode().replace(chr(0),' ')
 result['workers'][p.stem]=d
for p in sorted(root.glob('*/status.json'))+sorted(root.glob('evaluation/*/status.json'))+sorted(root.glob('training-probe-evaluation/*/status.json'))+sorted(root.glob('instruction-challenge/*/status.json'))+sorted(root.glob('final-evaluation/*/status.json')):
 d=json.loads(p.read_text());metrics=p.parent/'metrics.jsonl'
 if metrics.exists():
  rows=[json.loads(l) for l in metrics.read_text().splitlines()]
  def finite(v):
   if isinstance(v,dict):return all(finite(x) for x in v.values())
   if isinstance(v,list):return all(finite(x) for x in v)
   return not isinstance(v,float) or math.isfinite(v)
  d['all_logged_metrics_finite']=all(finite(x) for x in rows);d['last_metrics']=rows[-1] if rows else None
 for name in ['summary.json','final-identity.json']:
  if (p.parent/name).exists():d[name]=json.loads((p.parent/name).read_text())
 result['runs'][str(p.parent.relative_to(root))]=d
for p in sorted(root.glob('calibration/*/summary.json')):result['calibration'][p.parent.name]=json.loads(p.read_text())
for name in ['training-label-audit','holdout-image-audit','contrast-label-search','feature-extraction','long512-training','calibration3k','feature-extraction3k','long3k-training','ridge','ridge-validation','calibrated3k-quality-audit']:
 launch=root/(name+'-launch.json')
 if not launch.exists():continue
 pid=json.loads(launch.read_text())['pid'];live=(Path('/proc')/str(pid)).exists();summary=root/name/'summary.json'
 if name=='feature-extraction':summary=root/'frozen-A-features-v1/status.json'
 if name in ['long512-training','long3k-training','ridge']:summary=root/json.loads(launch.read_text())['name']/'status.json'
 if name=='ridge-validation':summary=root/'evaluation'/('validation-'+json.loads(launch.read_text())['name'])/'status.json'
 if name=='calibration3k':summary=root/'contrast-label-search3k/status.json'
 if name=='feature-extraction3k':summary=root/'frozen-A-features3k-v1/status.json'
 complete=summary.exists() and json.loads(summary.read_text()).get('status')=='completed'
 result['auxiliary'][name]={'pid':pid,'pid_live':live,'complete_report':complete,'unexpected_exit':not live and not complete}
print(json.dumps(result,allow_nan=False))
'''
def main():
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','SJTU1','python3 -'],input=REMOTE,text=True,capture_output=True,timeout=45)
    if r.returncode:raise RuntimeError(r.stderr)
    data=json.loads(r.stdout);out=Path(__file__).resolve().parent.parent/'metadata/goal-v2-20261103';out.mkdir(exist_ok=True)
    p=out/'latest-snapshot.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)
    print(json.dumps({'observed_at':data['observed_at'],'workers':{k:{f:v.get(f) for f in ['status','task','pid_live','child_pid_live']} for k,v in data['workers'].items()},
       'runs':{k:{f:v.get(f) for f in ['status','step','completed','all_logged_metrics_finite']} for k,v in data['runs'].items()},'calibration':list(data['calibration']),'auxiliary':data['auxiliary']},indent=2))
if __name__=='__main__':main()

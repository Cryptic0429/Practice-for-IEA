"""Final allocation watchdog; protect original service, permit useful final inference."""
import os,time,fcntl,signal,subprocess
from pathlib import Path
from functional_v1.common import EXP,read,write,stamp
from functional_v1.budget_watch import ours

def main():
 handle=(EXP/'budget-watch.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  targets=[];hard=False
  with (EXP/'resource.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);path=EXP/'resources.json';v=read(path);now=time.time()
   for j in v['jobs']:
    if 'ended_epoch' not in j and not ours(j):j.update(ended_epoch=now,ended_at=stamp(),status='terminal_process_missing',accounting_note='Conservative absent-process endpoint; no synthetic budget adjustment.')
   used=sum((j.get('ended_epoch',now)-j['started_epoch'])/3600 for j in v['jobs']);v['consumed_gpu_hours']=used;v['watchdog_updated_at']=stamp()
   if used>=v['limit_gpu_hours']-.0008:
    write(EXP/'BUDGET_STOP.json',dict(status='experiment_gpu_budget_stop',time=stamp(),occupied_gpu_hours_at_stop=used,hard_limit_gpu_hours=v['limit_gpu_hours'],buffer_gpu_hours=.0008,policy='Final <=2.88aggregate-second inference-exit reserve. No idle budget burn; original GPU1 service excluded.'))
    targets=[j for j in v['jobs'] if 'ended_epoch' not in j and ours(j)];hard=used>=v['limit_gpu_hours']
   write(path,v)
  for j in targets:
   assert j['gpu'] in ('0','2','3') and j['pid']!=3378799
   try:os.kill(j['pid'],signal.SIGKILL if hard else signal.SIGTERM)
   except ProcessLookupError:pass
  time.sleep(.1 if used>=23.95 else 1)
if __name__=='__main__':main()

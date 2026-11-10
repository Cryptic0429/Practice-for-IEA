"""Seal the real occupied-GPU ledger and shut down its now-idle watchdog."""
import os,signal,subprocess,fcntl
from pathlib import Path
from collections import defaultdict
from .common import *
from functional_v1.common import EXP as BUDGET

def main():
 assert not (EXP/'final-resource-summary.json').exists();v=read(BUDGET/'resources.json');assert all('ended_epoch' in j for j in v['jobs']);used=GPUJob.consumed(v);assert 23.999<=used<=24 and (BUDGET/'BUDGET_STOP.json').exists()
 h=read(EXP/'watchdog-handoff.json');pid=h['new_pid'];proc=Path('/proc')/str(pid);assert (proc/'cwd').resolve()==ROOT.resolve() and b'functional_v23.budget_watch' in (proc/'cmdline').read_bytes() and pid!=3378799
 os.kill(pid,signal.SIGTERM)
 with (BUDGET/'budget-watch.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX)
 v=read(BUDGET/'resources.json');assert GPUJob.consumed(v)==used
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines();mem={int(i):int(m) for i,m in (x.split(',') for x in gpu)};assert all(mem[i]<512 for i in (0,2,3))
 original=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split();assert '3378799' in original
 categories=defaultdict(float);jobs=[]
 for j in v['jobs']:
  s=j['stage'];key=None
  if s.startswith('v22-'):key='R22_input_features'
  elif s.startswith('interactive-trial-functional-v22-'):key='R22_trial_including_load_idle_failed_start'
  elif s.startswith('v23-direct-quality-'):key='R23_training'
  elif s.startswith('v23-fresh-evaluation-'):key='R23_fixed_development_inference'
  elif s.startswith('interactive-trial-functional-v23-'):key='R23_trial'
  elif s=='v23-final-export-robustness-inference':key='R23_export_stress'
  if key:categories[key]+=(j['ended_epoch']-j['started_epoch'])/3600;jobs.append(j)
 before=used-sum(categories.values());assert abs(before-23.26515351692836)<1e-8
 write(EXP/'final-resource-summary.json',dict(at=stamp(),status='24_hour_allocation_exhausted_with_exit_margin',exact_occupied_GPU_hours=used,display_GPU_hours=round(used,2),limit_GPU_hours=24,unused_exit_margin_seconds=(24-used)*3600,baseline_through_R21_GPU_hours=before,increment_categories_GPU_hours=dict(categories),new_jobs=jobs,total_ledger_jobs=len(v['jobs']),ledger_sha256=sha(BUDGET/'resources.json'),budget_stop_sha256=sha(BUDGET/'BUDGET_STOP.json'),original_GPU1_PID=3378799,physical_GPU_memory_MiB_after=mem,watchdog_exited_after_final_stop=True,remaining_new_inference_enabled=False,accounting='Sum of actual occupied per-GPU job intervals; includes model loading and trial waiting, not CPU-only label/training/render time. No idle padding or fabricated exact24.'))
 print('Actual GPU hours',used,'remaining cleanup margin',(24-used)*3600,'seconds; GPU1 preserved')
if __name__=='__main__':main()

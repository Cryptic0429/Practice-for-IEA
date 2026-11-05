"""Read-only live observation; never launches or restarts any workload."""
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import subprocess
from stage2_budget import ROOT,ledger


def main():
    result=ledger()
    queue=ROOT/'logs/stage2-A-edit-grpo-v1/corrected-preflight-queue.json'
    if queue.exists():
        state=json.loads(queue.read_text())
        proc=Path('/proc')/str(state['pid'])/'cmdline'
        cmd=proc.read_bytes() if proc.exists() else b''
        state['process_alive']=b'run_stage2_corrected_preflight.py' in cmd
        result['corrected_preflight_queue']=state
    acceptance=ROOT/'logs/stage2-A-edit-grpo-v1/corrected-preflight-acceptance.json'
    if acceptance.exists():
        result['corrected_preflight_acceptance']=json.loads(acceptance.read_text())
    pipeline=ROOT/'logs/stage2-A-edit-grpo-v1/experiment-pipeline.json'
    if pipeline.exists():
        state=json.loads(pipeline.read_text())
        proc=Path('/proc')/str(state['pid'])/'cmdline'
        cmd=proc.read_bytes() if proc.exists() else b''
        state['process_alive']=b'run_stage2_experiment.py' in cmd
        result['experiment_pipeline']=state
    result['gpu_snapshot']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,temperature.gpu,utilization.gpu',
                                                   '--format=csv'],text=True)
    active=[]
    for item in result['tasks']:
        if not item.get('process_alive'):
            continue
        out=Path(item['path']).parent
        entry=dict(run=out.name,pid=item['pid'],elapsed_seconds=item['elapsed_seconds'])
        status=json.loads(Path(item['path']).read_text())
        entry.update(status=status.get('status'),step=status.get('step'),completed=status.get('completed'),
                     status_updated_at=status.get('updated_at'))
        for key in ['candidates','metrics']:
            p=out/f'{key}.jsonl'
            if p.exists():
                contents=p.read_text()
                # Observe only newline-terminated records while writers append.
                complete=contents.rsplit('\n',1)[0] if '\n' in contents else ''
                rows=[json.loads(x) for x in complete.splitlines()]
                entry[key+'_rows']=len(rows)
                entry[key+'_mtime']=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()
                if key=='metrics' and rows:
                    entry['last_metrics']=rows[-1]
                    assert all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v,(float,int)))
                if key=='candidates':
                    assert all(math.isfinite(r['reward']) for r in rows)
                    entry['valid_candidates']=sum(r['parsing']['valid'] for r in rows)
        gpu=out/'gpu.csv'
        if gpu.exists() and gpu.stat().st_size:
            lines=gpu.read_text().splitlines()
            entry['last_gpu_sample']=lines[-1]
            entry['sampled_device_peak_mib']=max(int(s.split(',')[1]) for s in lines)
        active.append(entry)
    result['active_runs']=active
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()

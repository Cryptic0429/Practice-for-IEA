"""Read-only resource ledger from explicit run records and actual process liveness."""
from datetime import datetime,timezone
import json
from pathlib import Path

ROOT=Path('/mnt/sdc/zhc/IEA')
TERMINAL={'failed','completed','completed_training','awaiting_independent_reload','budget_stopped'}


def ledger():
    now=datetime.now(timezone.utc)
    tasks=[]
    for p in sorted((ROOT/'outputs/stage2-A-edit-grpo-v1').rglob('status.json')):
        s=json.loads(p.read_text())
        if 'started_at' not in s:
            continue
        start=datetime.fromisoformat(s['started_at'])
        proc=Path('/proc')/str(s.get('pid',0))
        cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace') if (proc/'cmdline').exists() else ''
        alive='stage2' in cmd and 'python' in cmd
        elapsed=s.get('elapsed_seconds')
        if s['status'] not in TERMINAL:
            if alive:
                elapsed=(now-start).total_seconds()
            elif elapsed is None:
                # Unknown unobserved termination: cannot safely grant remaining budget.
                elapsed=None
        if elapsed is None and s.get('updated_at'):
            elapsed=(datetime.fromisoformat(s['updated_at'])-start).total_seconds()
        tasks.append(dict(path=str(p),status=s['status'],pid=s.get('pid'),process_alive=alive,
                          elapsed_seconds=elapsed,gpu_hours=elapsed/3600 if elapsed is not None else None,
                          accounting='Allocated wall time on one GPU, including setup and validation waits'))
    # Reload verification is a separate GPU process, not included in training elapsed.
    for p in sorted((ROOT/'outputs/stage2-A-edit-grpo-v1').rglob('reload-verification.json')):
        if list(p.parent.glob('reload-attempt-*/status.json')):
            continue  # Already accounted from live/success/failed attempt records.
        s=json.loads(p.read_text())
        tasks.append(dict(path=str(p),status=s['status'],process_alive=False,
                          gpu_hours=s.get('gpu_hours'),elapsed_seconds=s.get('elapsed_seconds')))
    unknown=[t for t in tasks if t['gpu_hours'] is None or
             (t['status'] not in TERMINAL|{'passed'} and not t['process_alive'])]
    used=sum(t['gpu_hours'] or 0 for t in tasks)
    return dict(observed_at=now.isoformat(),total_gpu_hours=used,remaining_gpu_hours=max(0,16-used),
                unknown_intervals=unknown,budget_safe_to_launch=not unknown and used<16,tasks=tasks)


if __name__=='__main__':
    print(json.dumps(ledger(),indent=2))

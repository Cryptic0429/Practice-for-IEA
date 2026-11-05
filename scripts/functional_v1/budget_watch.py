"""CPU-only watchdog for the experiment's owned GPU PIDs, with a stop buffer."""
import fcntl
import json
import os
import re
from pathlib import Path
import signal
import time
from .common import EXP, read, write, stamp


def ours(job):
    try:
        path = Path('/proc')/str(job['pid'])
        command = (path/'cmdline').read_bytes().replace(b'\0', b' ').decode()
        # All numbered experimental modules share this same16 GPU-hour ledger.
        # PID/start-time checks below still prevent signalling unrelated processes.
        if not re.search(r'(?:^|\s)functional_v[1-9][0-9]*\.', command):
            return False
        tail = (path/'stat').read_text().rsplit(')', 1)[1].split()
        if tail[0] == 'Z':
            return False
        boot = next(float(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime '))
        created = boot + float(tail[19])/os.sysconf('SC_CLK_TCK')
        return job['started_epoch']-120 <= created <= job['started_epoch']+2
    except (OSError, ValueError, StopIteration):
        return False


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    handle = (EXP/'budget-watch.lock').open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        targets = []
        with (EXP/'resource.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = EXP/'resources.json'
            if path.exists():
                value = read(path)
                now = time.time()
                for job in value['jobs']:
                    if 'ended_epoch' not in job and not ours(job):
                        job.update(ended_epoch=now, ended_at=stamp(), status='terminal_process_missing',
                                   accounting_note='Conservative endpoint when the owned process was confirmed absent; no live task was restarted.')
                hours = sum((j.get('ended_epoch', now)-j['started_epoch'])/3600 for j in value['jobs'])
                value['consumed_gpu_hours'] = hours
                value['watchdog_updated_at'] = stamp()
                # Six aggregate GPU-minutes leave room for orderly model exit/checkpoint saving.
                if hours >= value['limit_gpu_hours']-.1:
                    write(EXP/'BUDGET_STOP.json', {'status': 'experiment_gpu_budget_stop', 'time': stamp(),
                        'occupied_gpu_hours_at_stop': hours, 'hard_limit_gpu_hours': value['limit_gpu_hours'],
                        'buffer_gpu_hours': .1})
                    targets = [j for j in value['jobs'] if 'ended_epoch' not in j and ours(j)]
                write(path, value)
        for job in targets:
            try:
                os.kill(job['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(5)


if __name__ == '__main__':
    main()

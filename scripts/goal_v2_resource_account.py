"""Descriptive union of GPU telemetry spans; not a claim of complete billing."""
import csv
import datetime as dt
import json
from pathlib import Path

ROOT = Path('/mnt/sdc/zhc/IEA')
OUT = ROOT / 'outputs/goal-v2-20261103'

def gpu_for(directory):
    for name in ['status.json', 'run.json']:
        p = directory / name
        if p.exists():
            d = json.loads(p.read_text())
            for item in [d, d.get('identity', {}), d.get('settings', {})]:
                if 'gpu' in item:
                    return str(item['gpu'])
    return None

def main():
    spans = []
    unassigned = []
    for p in sorted(OUT.rglob('gpu*.csv')):
        rows = list(csv.reader(p.open()))
        valid = []
        for row in rows:
            try:
                t = dt.datetime.strptime(row[0].strip(), '%Y/%m/%d %H:%M:%S.%f')
                t = t.replace(tzinfo=dt.timezone(dt.timedelta(hours=8))).astimezone(dt.timezone.utc)
                valid.append((t.timestamp(), float(row[1]), float(row[2])))
            except (ValueError, IndexError):
                continue
        if not valid:
            unassigned.append({'path': str(p.relative_to(ROOT)), 'reason': 'no parsed samples'})
            continue
        gpu = gpu_for(p.parent)
        item = {'path': str(p.relative_to(ROOT)), 'gpu': gpu, 'samples': len(valid),
                'start_utc': valid[0][0], 'end_utc': valid[-1][0],
                'sampled_span_seconds': valid[-1][0]-valid[0][0],
                'max_memory_mib': max(x[1] for x in valid),
                'max_temperature_c': max(x[2] for x in valid)}
        (spans if gpu is not None else unassigned).append(item)
    hours = {}
    for gpu in sorted({r['gpu'] for r in spans}):
        assert gpu in ['0', '2', '3'], 'Protected GPU1 must never appear'
        intervals = sorted((r['start_utc'], r['end_utc']) for r in spans if r['gpu'] == gpu)
        merged = []
        for a, b in intervals:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        hours[gpu] = sum(b-a for a, b in merged)/3600
    report = {'scope': 'Union of first-to-last sampled telemetry spans by GPU, including monitored loading/waiting; excludes gaps and unmonitored tasks.',
              'gpu_hours_covered_by_telemetry': hours, 'summed_gpu_hours_covered': sum(hours.values()),
              'telemetry_files': len(spans), 'unassigned': unassigned, 'spans': spans,
              'protected_gpu': 1, 'maximum_distinct_campaign_gpus': len(hours),
              'limits': ['Not a full resource/billing total: startup/teardown sample gaps, restore smoke and delivery replay have no telemetry span.',
                         'Host nvidia-smi timestamps parsed in verified CST UTC+8; archive work is CPU/I/O only.',
                         'Only campaign0/2/3 used; existing GPU1 demo retained.']}
    (OUT / 'resource-account.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'spans'}))

if __name__ == '__main__':
    main()

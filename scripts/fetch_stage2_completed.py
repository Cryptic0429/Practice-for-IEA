"""Download and audit one completed evaluation; never launch inference."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

REMOTE = r'''
import json,pathlib,math,hashlib,sys
job=sys.argv[1];split,model=job.rsplit('-',1)
root=pathlib.Path('/mnt/sdc/zhc/IEA')
folder=root/'outputs/stage2-A-edit-grpo-v1/evaluation'/job
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
status=read(folder/'status.json');summary=read(folder/'summary.json')
run=read(folder/'run.json');lock=read(root/'configs/stage2-selected-models.json')
source=root/'data/stage2/stage2-A-edit-grpo-v1'/f'{split}.json'
target=read(source)
rows=[json.loads(x) for x in (folder/'predictions.jsonl').read_text().splitlines()]
count=100 if split=='test_new' else 200
assert status['status']=='completed' and status['completed']==summary['rows']==len(rows)==count
assert [r['id'] for r in rows]==[r['id'] for r in target]
assert len({r['group_id'] for r in target})==count
assert summary['source_sha256']==sha(source)
assert summary['adapter_sha256']==run['identity']['adapter_sha256']==lock['models'][model]['adapter_sha256']
assert all(r['scoring']['input_readable'] and r['scoring']['reference_readable'] and not r['scoring']['metric_error'] for r in rows)
assert all(math.isfinite(r['scoring'][k][m]) for r in rows for k in ['baseline_metrics','edited_metrics'] for m in ['pixel_L1','pixel_L2'])
for key,kind,metric in [('mean_l1','edited_metrics','pixel_L1'),('mean_rmse','edited_metrics','pixel_L2'),('baseline_l1','baseline_metrics','pixel_L1')]:
    assert abs(sum(r['scoring'][kind][metric] for r in rows)/count-summary[key])<1e-12
images=[pathlib.Path(r['scoring']['edited_image_path']) for r in rows]
assert all(p.is_file() and p.stat().st_size>0 for p in images)
audit=dict(passed=True,job=job,rows=count,unique_source_groups=count,exact_id_order=True,
           frozen_source_and_adapter_identity=True,all_metrics_finite=True,
           all_images_exist=True,data_metric_errors=0,
           predictions_sha256=sha(folder/'predictions.jsonl'),
           scope='Complete job records, identities, aggregate metrics and image presence; full900comparison and image content audit remain separate.')
print(json.dumps(dict(files={n:(folder/n).read_text() for n in ['status.json','summary.json','run.json','predictions.jsonl']},audit=audit)))
'''


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('job', choices=[f'{s}-{m}' for s in ['test_new', 'test_regression']
                                    for m in ['A', 'SFT', 'GRPO']])
    args = cli.parse_args()
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                             'SJTU1', f'python3 - {args.job}'], input=REMOTE,
                            text=True, capture_output=True, check=True, timeout=45)
    data = json.loads(result.stdout)
    meta = Path(__file__).resolve().parents[1] / 'metadata/stage2-A-edit-grpo-v1'
    destination = meta / 'test-results' / args.job
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in data['files'].items():
        path = destination / name
        if path.exists():
            assert path.read_text() == content, f'Completed evidence changed: {path}'
        else:
            path.write_text(content)
    audit = destination / 'record-audit.json'
    if not audit.exists():
        data['audit']['checked_at'] = datetime.now(timezone.utc).isoformat()
        audit.write_text(json.dumps(data['audit'], indent=2) + '\n')
    progress_path = meta / 'progress.json'
    progress = json.loads(progress_path.read_text())
    progress.setdefault('completed_test_jobs', {})[args.job] = json.loads(audit.read_text())
    progress['last_goal_turn_classification'] = f'progress: {args.job} complete, audited and downloaded'
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(job=args.job, audit=data['audit'],
                          summary=json.loads(data['files']['summary.json']),
                          status=json.loads(data['files']['status.json'])), indent=2))


if __name__ == '__main__':
    main()

"""Local dependency-free writer for directly viewed anonymous review sheets."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def record(split, sheet_name, notes, bundle_name='first-epoch-and-baselines'):
    root = Path(__file__).resolve().parents[2]
    folder = root / 'experiments/functional-v7-20261105/review' / split
    bundle = folder / bundle_name
    manifest = json.loads((bundle / 'manifest.json').read_text())
    sheet = next(s for s in manifest['sheets'] if Path(s['sheet']).name == sheet_name)
    assert hashlib.sha256((root / sheet['sheet']).read_bytes()).hexdigest() == sheet['sha256']
    assert len(notes) == len(sheet['items'])
    items = {r['item_id']: r for r in manifest['items']}
    destination = folder / 'codex-observations.jsonl'
    old_text = destination.read_text() if destination.exists() else ''
    prior = {json.loads(line)['item_id'] for line in old_text.splitlines()}
    assert not prior.intersection(sheet['items']), 'Preserve previous raw observations'
    rows = []
    for item_id, (completion, quality, evidence) in zip(sheet['items'], notes):
        assert completion in ('C', 'P', 'F', 'U') and quality in ('A', 'm', 'M', 'U') and len(evidence) >= 20
        item = items[item_id]
        rows.append(dict(item_id=item_id, case_id=item['case_id'],
                         completion=dict(C='completed', P='partial', F='failed', U='uncertain')[completion],
                         quality=dict(A='acceptable', m='minor_issue', M='major_issue', U='uncertain')[quality],
                         evidence=evidence, image_sha256=item['image_sha256'], sheet=sheet['sheet'],
                         sheet_sha256=sheet['sha256'], bundle=bundle_name,
                         reviewer='Codex direct anonymous-image visual AI review; not human',
                         observed_at=dt.datetime.now(dt.timezone.utc).isoformat()))
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(old_text + ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    temporary.replace(destination)
    print('Recorded', sheet_name, ':', len(rows), 'anonymous observations; no model identities shown.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split')
    parser.add_argument('sheet_name')
    parser.add_argument('notes_json')
    parser.add_argument('--bundle', default='first-epoch-and-baselines')
    args = parser.parse_args()
    record(args.split, args.sheet_name, json.loads(args.notes_json), args.bundle)

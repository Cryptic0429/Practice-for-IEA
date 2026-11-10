"""Preserve direct observations before any new reward-model predictions."""
import datetime as dt
import hashlib
import json
from pathlib import Path


def record(case_id, evidence, good=('completed', 'acceptable'), opposite=('failed', 'acceptable')):
    root = Path(__file__).resolve().parents[2]
    source = root / 'data/training/functional-v8-20261106/calibration-candidates.json'
    manifest = json.loads(source.read_text())
    row = next(r for r in manifest['direction'] if r['id'] == case_id)
    sheet = next(r for r in manifest['sheets'] if r['id'] == case_id)
    assert hashlib.sha256((root / sheet['sheet']).read_bytes()).hexdigest() == sheet['sha256']
    assert len(evidence) >= 20
    destination = root / 'experiments/functional-v8-20261106/calibration-direct-observations.jsonl'
    prior = destination.read_text() if destination.exists() else ''
    assert case_id not in {r['id'] for r in map(json.loads, prior.splitlines())}
    note = dict(id=case_id, at=dt.datetime.now(dt.timezone.utc).isoformat(),
                input_sha256=row['input_sha256'], good_sha256=row['good_sha256'], bad_sha256=row['bad_sha256'],
                sheet=sheet, good=dict(completion=good[0], quality=good[1]),
                opposite=dict(completion=opposite[0], quality=opposite[1]),
                identity=dict(completion='failed', quality='input_unchanged'), evidence=evidence,
                scope='Direct full-frame AI comparison; original admitted label evidence retained, no unseen pixel-detail claim. Calibration roles visible, no reward outputs or model identities exist yet.',
                reviewer='Codex AI, not human')
    destination.write_text(prior + json.dumps(note, ensure_ascii=False) + '\n')
    print('Recorded calibration:', case_id)

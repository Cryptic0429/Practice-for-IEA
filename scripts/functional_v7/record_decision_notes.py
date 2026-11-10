"""Write observed decision/target checks for a complete visible candidate sheet."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def record(sheet_name, notes):
    root = Path(__file__).resolve().parents[2]
    folder = root / 'experiments/functional-v7-20261105/decision-candidates'
    manifest = json.loads((folder / 'manifest.json').read_text())
    sheet = next(s for s in manifest['sheets'] if Path(s['path']).name == sheet_name)
    assert set(sheet['case_ids']) == set(notes)
    assert hashlib.sha256((root / sheet['path']).read_bytes()).hexdigest() == sheet['sha256']
    target = folder / 'direct-observations.json'
    data = json.loads(target.read_text()) if target.exists() else dict(
        reviewer='Codex direct visual/request AI review; not human or blinded',
        policy='A decline returns no image. A diagnostic rejected edit is never a target or successful result. AI counterrequests never reclassify the earlier original request.', observations=[])
    assert not any(r['sheet'] == sheet['path'] for r in data['observations'])
    for cid, (completion, quality, evidence) in notes.items():
        assert completion in ('completed', 'appropriate_decline', 'partial', 'uncertain', 'failed')
        assert quality in ('acceptable', 'minor_issue', 'major_issue', 'uncertain', 'not_applicable') and evidence.strip()
        candidate = next(c for c in manifest['records'] if c['id'] == cid)
        assert (quality == 'not_applicable') == (candidate['action'] == 'decline')
        data['observations'].append(dict(case_id=cid, completion=completion, quality=quality, evidence=evidence,
                                        at=dt.datetime.now(dt.timezone.utc).isoformat(), sheet=sheet['path'],
                                        sheet_sha256=sheet['sha256'], admitted=False))
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(target)
    print('Recorded', sheet_name, '; observations:', len(data['observations']), '; no admission.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sheet_name')
    parser.add_argument('notes_json')
    args = parser.parse_args()
    record(args.sheet_name, json.loads(args.notes_json))

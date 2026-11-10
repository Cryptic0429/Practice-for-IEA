"""Append observations only after the listed native files were actually viewed."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'experiments/functional-v18-20261107'


def record(value):
    manifest = EXP / 'review/manifest.json'
    pair = next(p for p in json.loads(manifest.read_text())['items'] if p['pair_id'] == value['pair_id'])
    assert len(value['cases']) == 2
    path = EXP / 'native-observations.jsonl'
    prior = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    assert not any(r['pair_id'] == pair['pair_id'] for r in prior)
    observed = []
    lookup = {'CA': ('completed', 'acceptable'), 'Cm': ('completed', 'minor_issue'),
              'PA': ('partial', 'acceptable'), 'UA': ('uncertain', 'acceptable')}
    for case, supplied in zip(pair['cases'], value['cases']):
        assert case['case_id'] == supplied['case_id']
        labels = ['INPUT'] + [k for k, v in case['mapping'].items() if v['origin'] in ('own_label', 'crossed_label')]
        assets = [a for a in case['native'] if a['label'] in labels]
        assert len(assets) == 3
        for asset in assets:
            assert hashlib.sha256((ROOT / asset['path']).read_bytes()).hexdigest() == asset['sha256']
        grades = {}
        for origin in ('own_label', 'crossed_label'):
            completion, quality = lookup[supplied[origin]]
            grades[origin] = dict(completion=completion, quality=quality)
        observed.append(dict(case_id=case['case_id'], whole_native_assets_actually_viewed=assets,
            candidate_grades=grades, evidence=supplied['evidence']))
    row = dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), pair_id=pair['pair_id'],
        reviewer='Codex direct visual AI review; not human/user', cases=observed,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), view_scope='whole unscaled input and both own/crossed results',
        candidate_mapping_known_during_native_followup=True, prior_numeric_diagnostic_exposure=True,
        fully_blind_independent_review=False, automatic_training_admission=False)
    with path.open('a') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print('Recorded native observations:', pair['pair_id'])


if __name__ == '__main__':
    record(json.load(sys.stdin))

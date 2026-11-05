"""Record direct Stage 2 image observations before reading the private model order.

prepare/record/validate read only the public bundle. record accepts a JSON list
of manually authored cases, each with context_note and X/Y/Z panel notes.
No judgments are inferred from metrics, filenames, or model identities.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


LABELS = ('X', 'Y', 'Z')
FIELDS = {
    'direction': {'aligned', 'partly_aligned', 'opposite', 'no_visible_change', 'ambiguous'},
    'strength': {'reasonable', 'too_weak', 'too_strong', 'unclear'},
    'unrelated_changes': {'none_obvious', 'possible', 'clear', 'unclear'},
    'noop_or_failure': {'none_obvious', 'apparent_noop', 'visible_failure', 'unclear'},
}
SCOPE = 'AI-assisted qualitative inspection; not a controlled human blind study.'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, data):
    with path.open('x') as handle:
        handle.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def validate_case(case):
    assert isinstance(case['context_note'], str) and case['context_note'].strip()
    assert set(case['panels']) == set(LABELS)
    for note in case['panels'].values():
        assert set(note) == set(FIELDS) | {'observation'}
        for field, allowed in FIELDS.items():
            assert note[field] in allowed, (case['case'], field, note[field])
        assert isinstance(note['observation'], str) and note['observation'].strip()
    assert case['reviewed_at'] and len(case['reviewed_png_sha256']) == 64


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'record', 'validate', 'decode'])
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--renders', type=Path)
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--decoded', type=Path)
    args = parser.parse_args()
    manifest_path = args.bundle / 'manifest.json'
    manifest = read(manifest_path)
    cases = manifest['cases']
    assert manifest['role'] == 'blinded_test_visual_review'
    assert manifest['case_count'] == len(cases) == 30
    assert [c['case'] for c in cases] == list(range(1, 31))
    assert Counter(c['split'] for c in cases) == {'test_new': 20, 'test_regression': 10}
    assert len({(c['split'], c['id']) for c in cases}) == 30
    for case in cases:
        assert [p['label'] for p in case['images']] == ['Original', 'Reference', *LABELS]
        for panel in case['images']:
            assert sha(args.bundle / panel['path']) == panel['sha256']
    if args.mode == 'prepare':
        records = dict(scope=SCOPE, manifest_sha256=sha(manifest_path),
                       plan_sha256=manifest['plan_sha256'],
                       model_identities_hidden_during_recording=True,
                       category_definitions={k: sorted(v) for k, v in FIELDS.items()},
                       limitation='Output style may reveal identity. Visual no-op/failure is a judgment, not parser or pixel-equality evidence.',
                       cases=[dict(case=c['case'], split=c['split'], id=c['id'],
                                   instruction=c['instruction'], context_note='',
                                   reviewed_at='', reviewed_png_sha256='',
                                   panels={label: {field: '' for field in [*FIELDS, 'observation']}
                                           for label in LABELS}) for c in cases])
        write_new(args.records, records)
        print('Prepared 30 blank cases; no observations have been made.')
        return
    records = read(args.records)
    assert records['scope'] == SCOPE
    assert records['model_identities_hidden_during_recording'] is True
    assert records['manifest_sha256'] == sha(manifest_path)
    assert records['plan_sha256'] == manifest['plan_sha256']
    assert len(records['cases']) == 30
    for observed, case in zip(records['cases'], cases):
        for field in ('case', 'split', 'id', 'instruction'):
            assert observed[field] == case[field]
    if args.mode == 'record':
        assert args.batch and args.renders
        assert not args.records.with_suffix('.sealed.json').exists(), 'Already decoded; preserve observations.'
        batch = read(args.batch)
        assert isinstance(batch, list) and batch
        assert len({item['case'] for item in batch}) == len(batch)
        for item in batch:
            number = item['case']
            assert isinstance(number, int) and 1 <= number <= 30
            observed = records['cases'][number - 1]
            assert not observed['reviewed_at'], 'Do not overwrite existing observations.'
            assert item['directly_inspected_all_five_panels'] is True
            png = args.renders / f'case-{number:02d}.png'
            assert sha(png) == item['reviewed_png_sha256']
            observed.update(context_note=item['context_note'], panels=item['panels'],
                            reviewed_at=datetime.now(timezone.utc).isoformat(),
                            reviewed_png_sha256=sha(png))
            validate_case(observed)
        temporary = args.records.with_suffix('.tmp')
        temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(args.records)
        print('Recorded cases:', sum(bool(c['reviewed_at']) for c in records['cases']))
        return
    for observed in records['cases']:
        validate_case(observed)
    if args.mode == 'validate':
        print('All 30 cases and 90 anonymous model observations recorded.')
        return
    # The private plan is first opened only after every observation validates.
    assert args.plan and args.decoded
    assert sha(args.plan) == manifest['plan_sha256']
    plan = read(args.plan)
    assert len(plan['cases']) == 30
    decoded_cases = []
    by_model = {name: [] for name in ('A', 'SFT', 'GRPO')}
    for observed, private in zip(records['cases'], plan['cases']):
        for field in ('case', 'split', 'id'):
            assert observed[field] == private[field]
        assert private['anonymous_labels'] == list(LABELS)
        assert sorted(private['model_order']) == sorted(by_model)
        models = {}
        for label, model in zip(LABELS, private['model_order']):
            note = observed['panels'][label]
            models[model] = dict(anonymous_label=label, **note)
            by_model[model].append(note)
        decoded_cases.append(dict(case=observed['case'], split=observed['split'],
                                  id=observed['id'], context_note=observed['context_note'], models=models))
    result = dict(scope=SCOPE, decoded_at=datetime.now(timezone.utc).isoformat(),
                  observations_sha256=sha(args.records), manifest_sha256=sha(manifest_path),
                  plan_sha256=sha(args.plan), case_count=30, model_panel_count=90,
                  cases=decoded_cases,
                  descriptive_counts={m: {f: dict(Counter(n[f] for n in notes)) for f in FIELDS}
                                      for m, notes in by_model.items()})
    assert not args.decoded.exists()
    write_new(args.records.with_suffix('.sealed.json'),
              {k: result[k] for k in ('observations_sha256', 'manifest_sha256', 'plan_sha256', 'decoded_at')})
    write_new(args.decoded, result)
    print('Decoded 30 cases / 90 model panels; observations sealed.')


if __name__ == '__main__':
    main()

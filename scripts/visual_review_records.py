"""Prepare complete blinded records; decode identities only after observations exist."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/sdc/zhc/IEA')
ALLOWED = {
    'direction': {'aligned', 'partly_aligned', 'opposite', 'no_visible_change', 'ambiguous'},
    'strength': {'reasonable', 'too_weak', 'too_strong', 'unclear'},
    'unrelated_changes': {'none_obvious', 'possible', 'clear', 'unclear'},
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    assert not path.exists(), f'Preserve previous observations: {path}'
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def validate(observations, ids):
    assert observations['scope'] == 'AI-assisted qualitative inspection, not a human user study.'
    assert observations['model_identities_hidden_during_recording'] is True
    assert len(observations['cases']) == 30
    assert [case['id'] for case in observations['cases']] == ids
    expected = {f'Model {i}' for i in range(1, 6)}
    for index, case in enumerate(observations['cases'], 1):
        assert case['case'] == index and set(case['panels']) == expected
        assert case['context_note'].strip(), f'Context not reviewed: {index}'
        for panel, note in case['panels'].items():
            for field, allowed in ALLOWED.items():
                assert note[field] in allowed, f'Missing/invalid observation: {index} {panel} {field}'
            assert note['observation'].strip(), f'Missing visual evidence: {index} {panel}'


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('mode', choices=['prepare', 'validate', 'decode'])
    args = cli.parse_args()
    folder = ROOT / 'outputs/evaluation-v1'
    manifest = read(folder / 'blinded-visual-bundle/manifest.json')
    assert manifest['role'] == 'blinded_test_visual_review' and manifest['case_count'] == 30
    protocol = read(ROOT / 'configs/evaluation-protocol-v1.json')
    ids_path = Path(protocol['visual_ids_path'])
    assert sha(ids_path) == protocol['visual_ids_sha256']
    ids = read(ids_path)
    assert [case['id'] for case in manifest['cases']] == ids
    path = folder / 'visual-review-observations.json'
    if args.mode == 'prepare':
        # Blank records must be filled from direct image inspection, never from metrics.
        cases = [dict(case=c['case'], id=c['id'], instruction=c['instruction'], context_note='',
                      panels={f'Model {i}':dict(direction='', strength='', unrelated_changes='', observation='')
                              for i in range(1, 6)}) for c in manifest['cases']]
        write(path, dict(scope='AI-assisted qualitative inspection, not a human user study.',
                         model_identities_hidden_during_recording=True,
                         protocol_sha256=manifest['protocol_sha256'],
                         category_definitions={k:sorted(v) for k,v in ALLOWED.items()},
                         limitation='Panel numbering conceals labels, but output style can reveal identity. No claim of a controlled blinded human trial.',
                         cases=cases))
        print('BLANK_OBSERVATION_RECORDS_PREPARED', path)
        return
    observations = read(path)
    validate(observations, ids)
    if args.mode == 'validate':
        print('ALL_30_CASES_AND_150_MODEL_PANELS_RECORDED')
        return
    key_path = ids_path.with_name('visual-blinding-key.json')
    assert sha(key_path) == protocol['visual_blinding_key_sha256']
    key = read(key_path)
    assert [case['id'] for case in key] == ids
    assert not (folder / 'visual-review-decoded.json').exists()
    # This saved hash makes later observation edits detectable.
    observation_sha = sha(path)
    by_model = {m:[] for m in protocol['models']}
    cases = []
    for observed, mapping in zip(observations['cases'], key):
        decoded = {}
        for panel, note in observed['panels'].items():
            model = mapping['panels'][panel]
            decoded[model] = dict(panel=panel, **note)
            by_model[model].append(dict(case=observed['case'], id=observed['id'], **note))
        cases.append(dict(case=observed['case'], id=observed['id'], models=decoded))
    counts = {m:{field:dict(Counter(note[field] for note in notes)) for field in ALLOWED}
              for m,notes in by_model.items()}
    write(folder / 'visual-review-decoded.json',
          dict(decoded_at=datetime.now(timezone.utc).isoformat(), observations_sha256=observation_sha,
               scope=observations['scope'], case_count=30, panel_count=150, descriptive_counts=counts,
               cases=cases, limitation=observations['limitation']))
    print(json.dumps(dict(case_count=30, panel_count=150, observations_sha256=observation_sha,
                         descriptive_counts=counts), indent=2))


if __name__ == '__main__':
    main()

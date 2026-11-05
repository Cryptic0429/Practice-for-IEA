"""Append an actual anonymous visual observation; freeze only after all30x7."""
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'artifacts/goal-v2-20261103/blind-review'
PATH = DIRECTORY / 'observations-before-decoding.json'

def main():
    payload = json.loads(sys.argv[1])
    document = json.loads(PATH.read_text())
    assert document['status'] == 'in_progress'
    assert not document['model_identities_decoded']
    manifest = json.loads((DIRECTORY / 'manifest.json').read_text())
    assert hashlib.sha256((DIRECTORY / 'manifest.json').read_bytes()).hexdigest() == document['manifest_sha256']
    case_number = payload['case']
    assert case_number not in {row['case'] for row in document['cases']}
    spec = next(row for row in manifest['cases'] if row['case'] == case_number)
    assert len(payload['ratings']) == 7
    names = ['instruction', 'strength', 'unrelated_change', 'reference_match_relative_input']
    outputs = []
    for index, values in enumerate(payload.pop('ratings'), 1):
        assert len(values) == 5
        item = {'output': f'Output{index}'}
        for field, value in zip(names, values[:4]):
            assert value in document['rubric'][field], (field, value)
            item[field] = value
        item['note'] = values[4]
        assert len(item['note']) >= 12
        outputs.append(item)
    payload.update(id=spec['id'], group_id=spec['group_id'], instruction=spec['instruction'],
                   observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), outputs=outputs)
    document['cases'].append(payload)
    document['cases'].sort(key=lambda row: row['case'])
    if len(document['cases']) == 30:
        assert {row['case'] for row in document['cases']} == set(range(1, 31))
        document['status'] = 'complete_before_decoding'
        document['completed_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    temporary = PATH.with_suffix('.tmp')
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(PATH)
    print(json.dumps({'saved_case': case_number, 'cases_recorded': len(document['cases']),
                      'status': document['status'], 'sha256': hashlib.sha256(PATH.read_bytes()).hexdigest()}))

if __name__ == '__main__':
    main()

"""Persist direct visual observations without selecting a training label."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def record(case_id, notes, folder_name='label-repairs'):
    root = Path(__file__).resolve().parents[2]
    assert folder_name in ('label-repairs', 'label-quality-repairs')
    folder = root / 'experiments/functional-v7-20261105' / folder_name
    manifest = json.loads((folder / 'manifest.json').read_text())
    sheet = next(s for s in manifest['sheets'] if s['case_id'] == case_id)
    candidates = [r for r in manifest['records'] if r['case_id'] == case_id]
    expected = {r.get('label', 'new') for r in candidates}
    if len(expected) == 1 and set(notes) == {'new'}:
        notes = {next(iter(expected)): notes['new']}
    assert set(notes) == expected, 'Cover exactly the visible candidate labels'
    assert hashlib.sha256((root / sheet['sheet']).read_bytes()).hexdigest() == sheet['sha256']
    target = folder / 'direct-observations.json'
    data = json.loads(target.read_text()) if target.exists() else dict(
        reviewer='Codex direct visual AI review; not human',
        policy='No automatic admission. Input revisions explicitly disclosed; old failed tasks remain failed. Full-size followup is required for uncertain quality.',
        observations=[])
    assert not any(r['case_id'] == case_id for r in data['observations']), 'Preserve prior observations'
    for label, (completion, quality, evidence) in notes.items():
        assert completion in ('C', 'P', 'F', 'U')
        assert quality in ('A', 'm', 'M', 'U') and evidence.strip()
        exposed = folder_name == 'label-repairs' and case_id == 'train-045-restore-exposure-strong'
        blind = sheet['kind'] == 'parameter_repair' and not exposed
        data['observations'].append(dict(
            case_id=case_id, label=label, at=dt.datetime.now(dt.timezone.utc).isoformat(),
            sheet=sheet['sheet'], sheet_sha256=sheet['sha256'],
            completion=dict(C='completed', P='partial', F='failed', U='uncertain')[completion],
            quality=dict(A='acceptable', m='minor_issue', M='major_issue', U='uncertain')[quality],
            evidence=evidence, candidate_parameter_mapping_hidden=blind,
            disclosure=('First parameter pair was printed before visual review.' if exposed
                        else 'Synthetic input revision is explicitly visible.' if not blind else 'Candidate parameter mapping hidden.'),
            inspection='Full-frame panel; no pixel-detail claim beyond visible scale.', admitted=False))
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(target)
    print('Recorded', case_id, '; total', len(data['observations']), '; no admission.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case_id')
    parser.add_argument('notes_json')
    parser.add_argument('--folder', default='label-repairs')
    args = parser.parse_args()
    record(args.case_id, json.loads(args.notes_json), args.folder)

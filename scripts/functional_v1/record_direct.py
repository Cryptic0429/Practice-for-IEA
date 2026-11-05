"""Record observations already made by the agent; never generates review ratings."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from .common import ROOT, DATA, EXP, read, write, append, sha, stamp


def case_digest(case):
    fields = {k: case[k] for k in ('id','image','input_sha256','instructions','expected_change','must_preserve')}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('observations', type=Path)
    ap.add_argument('--revision', action='store_true')
    args = ap.parse_args()
    root = EXP/'review/training'
    cases = {c['id']:c for c in read(DATA/'train-cases.json')}
    proposals = {r['case_id']:r for r in read(root/'direct-proposals.json')}
    by_case = {c['case_id']:s for s in read(root/'sheet-manifest.json') for c in s['cases']}
    output = root/'codex-observations.jsonl'
    previous = [json.loads(l) for l in output.read_text().splitlines()] if output.exists() else []
    seen = {r['case_id'] for r in previous}
    (root/'evidence').mkdir(exist_ok=True)
    for note in read(args.observations):
        case_id = note['case_id']
        if case_id in seen and not args.revision:
            raise ValueError('Existing observation; an explicit revision is required: '+case_id)
        assert note['completion'] in ('completed','partial','failed','uncertain')
        assert note['quality'] in ('acceptable','minor_issue','major_issue','uncertain')
        assert len(note.get('evidence','')) >= 25, 'A concrete visible observation is required'
        target = proposals[case_id]['selected']
        sheet = by_case[case_id]
        path = ROOT/sheet['sheet']
        if not path.exists():
            path = root/'sheets'/path.name
        assert sha(path) == sheet['sha256']
        assert next(c for c in sheet['cases'] if c['case_id']==case_id)['image_sha256'] == target['sha256']
        evidence = root/'evidence'/(sheet['sha256']+'.jpg')
        if not evidence.exists():
            shutil.copy2(path, evidence)
        record = {**note, 'reviewer': 'Codex direct visual review; AI-assisted, not human',
                  'case_spec_sha256': case_digest(cases[case_id]),
                  'image_sha256': target['sha256'], 'sheet_sha256': sheet['sha256'],
                  'sheet_evidence': str(evidence.relative_to(ROOT)), 'observed_at': stamp()}
        append(output, record)
        previous.append(record)
        seen.add(case_id)
    latest = {r['case_id']:r for r in previous}
    write(root/'direct-review-status.json', {'status': 'in_progress', 'viewed_unique': len(latest),
        'accepted': sum(r['completion']=='completed' and r['quality']=='acceptable' for r in latest.values()),
        'completion_counts': dict(__import__('collections').Counter(r['completion'] for r in latest.values())),
        'total': len(cases), 'human_review': False, 'updated_at': stamp()})


if __name__ == '__main__':
    main()

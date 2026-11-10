"""Persist a Codex full-resolution followup; does not infer visual judgments."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def record(case_id, completion, quality, evidence):
    root=Path(__file__).resolve().parents[2]
    folder=root/'experiments/functional-v7-20261105/soft-label-candidates'
    candidate=next(c for c in json.loads((folder/'manifest.json').read_text())['cases'] if c['case_id']==case_id)
    for path,digest in [(candidate['input_image'],candidate['input_sha256']),
                        (candidate['new_candidate'],candidate['new_candidate_sha256'])]:
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest
    target=folder/'full-resolution-observations.json'
    data=json.loads(target.read_text())
    assert not any(r['case_id']==case_id for r in data['observations']), 'Preserve previous observations'
    assert completion in ('completed','partial','failed','uncertain')
    assert quality in ('acceptable','minor_issue','major_issue','uncertain') and evidence.strip()
    data['observations'].append(dict(
        case_id=case_id,at=dt.datetime.now(dt.timezone.utc).isoformat(),completion=completion,quality=quality,evidence=evidence,
        image=candidate['new_candidate'],image_sha256=candidate['new_candidate_sha256'],
        input_image=candidate['input_image'],input_sha256=candidate['input_sha256'],
        reviewer='Codex direct full-resolution input and result visual AI review after role lookup; not human or blinded',
        new_training_admission=False))
    temp=target.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(target)
    print('Saved full-resolution observation:',case_id)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('case_id');ap.add_argument('completion');ap.add_argument('quality');ap.add_argument('evidence')
    args=ap.parse_args();record(args.case_id,args.completion,args.quality,args.evidence)

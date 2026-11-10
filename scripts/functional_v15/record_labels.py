"""Dependency-free recording of actually viewed candidates; no automatic admission."""
from pathlib import Path
import json,hashlib,datetime

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'experiments/functional-v15-20261106'


def record(stage,cid,completion,quality,evidence,previous_observation=''):
    assert stage in ('labels','quality') and completion in 'CPFU' and quality in 'AmMU' and len(evidence)>30
    folder=EXP/('label-candidates' if stage=='labels' else 'quality-challenges')
    manifest=folder/'manifest.json';item=next(r for r in json.loads(manifest.read_text())['items'] if r['id']==cid)
    criteria_file=EXP/'functional-review-criteria.json';criteria=next(c for c in json.loads(criteria_file.read_text())['items'] if c['id']==cid)
    assert item['instructions']==criteria['original_requests_unchanged']
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(ROOT/item['panel']['path'])==item['panel']['sha256']
    destination=EXP/('label-observations.jsonl' if stage=='labels' else 'quality-challenge-observations.jsonl')
    previous=destination.read_text() if destination.exists() else '';assert cid not in {r['id'] for r in map(json.loads,previous.splitlines())}
    row=dict(id=cid,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,
        completion=completion,quality=quality,evidence=evidence,previous_tool_observation=previous_observation,
        panel=item['panel'],source_input_sha256=item['input']['sha256'],candidate_image_sha256=item['candidate']['sha256'],
        manifest_sha256=digest(manifest),criteria_sha256=digest(criteria_file),parameters=item['parameters'],
        reviewer='Codex direct candidate-image AI review; not human or user approval',
        original_size_followup_required=quality=='U',training_label_admitted=False)
    temp=destination.with_suffix('.tmp');temp.write_text(previous+json.dumps(row,ensure_ascii=False)+'\n');temp.replace(destination)
    print('Recorded',stage,cid,completion,quality,'; no label automatically admitted')

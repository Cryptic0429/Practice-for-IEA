"""Preserve each actual training-candidate observation with exact image provenance."""
import datetime as dt
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'experiments/functional-v11-20261106'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def record(cid,completion,quality,evidence,full=False):
    source=EXP/'label-candidates/manifest.json';manifest=json.loads(source.read_text())
    c=next(c for c in manifest['negative_curve_candidates']+manifest['cool_cast_candidates'] if c['id']==cid)
    assert completion in ('C','P','F','U') and quality in ('A','m','M','U') and len(evidence)>30
    for i in (c['input'],c['candidate'],c['panel']):assert sha(ROOT/i['path'])==i['sha256']
    target=EXP/'label-candidates/direct-observations.jsonl';old=target.read_text() if target.exists() else ''
    assert cid not in {json.loads(line)['id'] for line in old.splitlines()},'Keep raw notes immutable'
    row=dict(id=cid,at=dt.datetime.now(dt.timezone.utc).isoformat(),completion=dict(C='completed',P='partial',F='failed',U='uncertain')[completion],
        quality=dict(A='acceptable',m='minor_issue',M='major_issue',U='uncertain')[quality],evidence=evidence,
        input=c['input'],candidate=c['candidate'],panel=c['panel'],manifest_sha256=sha(source),
        original_size_result_also_viewed=bool(full),reviewer='Codex direct candidate-image AI review, not human or model capability assessment',training_admitted=False)
    target.write_text(old+json.dumps(row,ensure_ascii=False)+'\n')
    print(cid,completion,quality)

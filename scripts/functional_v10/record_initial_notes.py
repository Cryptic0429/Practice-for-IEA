"""Preserve directly observed initial candidates; no final label admission."""
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'experiments/functional-v10-20261106'

def record(cid,deep,shallow,noise):
    manifest=EXP/'counterfactual-candidates/manifest.json'
    c=next(c for c in json.loads(manifest.read_text())['cases'] if c['id']==cid)
    sheet=ROOT/c['sheet'];assert hashlib.sha256(sheet.read_bytes()).hexdigest()==c['sheet_sha256']
    path=EXP/'counterfactual-candidates/direct-observations.jsonl'
    old=path.read_text() if path.exists() else ''
    assert cid not in {json.loads(l)['id'] for l in old.splitlines()}
    row=dict(id=cid,at=dt.datetime.now(dt.timezone.utc).isoformat(),sheet=c['sheet'],sheet_sha256=c['sheet_sha256'],
        clean_deep=deep,clean_shallow=shallow,initial_noise_attempts=noise,
        reviewer='Codex direct visual AI review; not human',final_admission=False,
        pending='Compensated severe-noise alternatives, mild-noise/compound/darkening cases and needed original-size followups remain to be checked. Proposed refusal not yet admitted.')
    path.write_text(old+json.dumps(row,ensure_ascii=False)+'\n')

if __name__=='__main__':
    record('visual-risk-01',
        dict(completion='completed',quality='acceptable',evidence='The child, plaid shirt, blue vest and wooden rails are substantially brighter from the deep-dark input while their colour and structure remain natural.'),
        dict(completion='completed',quality='acceptable',evidence='The shallow-dark input receives a clearly visible lift with natural skin, plaid colour and rail structure; it is slightly brighter than the canonical scene without obvious new clipping.'),
        dict(completion='completed',quality='major_issue',evidence='All three initial brightening attempts make large red-green chroma blocks much more conspicuous across the face, clothing and vegetation. Compensated alternatives remain unreviewed, so no refusal label is admitted.'))
    record('visual-risk-02',
        dict(completion='completed',quality='acceptable',evidence='The portrait, sari folds and leaves are visibly brighter from the deep-dark condition while skin and dark clothing retain recognizable natural layers.'),
        dict(completion='completed',quality='acceptable',evidence='The shallow-dark portrait receives a visible lift with plausible skin and foliage colour and the sari pattern intact; no conspicuous severe new highlight loss is evident in the panel.'),
        dict(completion='completed',quality='major_issue',evidence='All three initial attempts expose strong coloured horizontal bands and mottling across face, arm, sari and foliage. This supports a quality concern, but compensation attempts must be reviewed before any decision admission.'))

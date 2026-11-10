"""Record directly viewed feasibility attempts; no automatic refusal admission."""
import datetime as dt
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];EXP=ROOT/'experiments/functional-v11-20261106'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def record(group,notes,details=False,rock=False):
    source=EXP/'quality-revalidation/manifest.json';m=json.loads(source.read_text())
    sheet=next(s for s in m['panels'] if s['group']==group)
    assert len(notes)==len(sheet['candidates']) and sha(ROOT/sheet['panel']['path'])==sheet['panel']['sha256']
    lookup={r['id']:r for r in m['attempts']};target=EXP/'quality-revalidation/direct-observations.jsonl'
    prior=target.read_text() if target.exists() else '';seen={json.loads(line)['id'] for line in prior.splitlines()}
    rows=[]
    for rid,(completion,quality,evidence) in zip(sheet['candidates'],notes):
        assert rid not in seen and completion in ('C','P','F','U') and quality in ('A','m','M','U') and len(evidence)>30
        r=lookup[rid]
        assert sha(ROOT/r['input']['path'])==r['input']['sha256'] and sha(ROOT/r['output']['path'])==r['output']['sha256']
        supplements=[]
        for folder,enabled in [('details',details),('details-rock',rock)]:
            if enabled:
                p=EXP/'quality-revalidation'/folder/'manifest.json';d=json.loads(p.read_text())
                items=[i for i in d['items'] if i['role'] in ('INPUT',rid) and (folder=='details-rock' or i['case_id']==r['case_id'])]
                assert len(items)==2
                for i in items:assert sha(ROOT/i['path'])==i['sha256']
                supplements.append(dict(manifest=str(p.relative_to(ROOT)),manifest_sha256=sha(p),images=items))
        rows.append(dict(id=rid,case_id=r['case_id'],group=group,at=dt.datetime.now(dt.timezone.utc).isoformat(),
            completion=dict(C='completed',P='partial',F='failed',U='uncertain')[completion],quality=dict(A='acceptable',m='minor_issue',M='major_issue',U='uncertain')[quality],
            evidence=evidence,input=r['input'],output=r['output'],panel=sheet['panel'],source_manifest_sha256=sha(source),original_pixel_supplements=supplements,
            reviewer='Codex direct feasibility-candidate visual AI review, not human',training_admitted=False))
    target.write_text(prior+''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print('Recorded',group,len(rows))

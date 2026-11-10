"""Save only actually viewed anonymous diagnostic observations."""
import datetime,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'experiments/functional-v17-20261107'


def record(cid,label,stage,completion,quality,evidence):
    assert label in ('A','B') and stage in ('overview','native')
    assert completion in 'CPFU' and quality in 'AmMU' and len(evidence)>30
    path=EXP/'review/manifest.json';item=next(i for i in json.loads(path.read_text())['items'] if i['case_id']==cid)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(ROOT/item['sheet'])==item['sheet_sha256']
    assets=[i for i in item['native'] if i['label'] in ('INPUT',label)]
    assert all(sha(ROOT/i['path'])==i['sha256'] for i in assets)
    dest=EXP/'observations.jsonl';old=dest.read_text() if dest.exists() else ''
    assert (cid,label,stage) not in {(r['case_id'],r['label'],r['stage']) for r in map(json.loads,old.splitlines())}
    row=dict(case_id=cid,label=label,stage=stage,completion=completion,quality=quality,evidence=evidence,
        at=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_sha256=sha(path),sheet=item['sheet'],
        sheet_sha256=item['sheet_sha256'],assets=assets,reviewer='Codex direct anonymous-image AI review; not human/user feedback',
        scope='Four reused TRAIN sources only; no training, admission, heldout result or generalization claim.')
    dest.write_text(old+json.dumps(row,ensure_ascii=False)+'\n')
    print('Recorded',cid,label,stage,completion,quality)

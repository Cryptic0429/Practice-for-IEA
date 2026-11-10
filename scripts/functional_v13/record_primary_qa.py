"""Append native-pixel supplements without replacing tentative sheet notes."""
import json,hashlib,datetime
from pathlib import Path


def record(bundle,notes,observations='primary-qa-observations.jsonl'):
    root=Path(__file__).resolve().parents[2];exp=root/'experiments/functional-v13-20261106'
    path=exp/'primary-qa'/bundle/'manifest.json';m=json.loads(path.read_text());bank={r['id']:r for r in m['items']}
    dest=exp/observations;old=dest.read_text() if dest.exists() else ''
    done={json.loads(l)['id'] for l in old.splitlines()};rows=[]
    for sid,completion,quality,evidence in notes:
        assert sid not in done and completion in ('C','P','F','U') and quality in ('A','m','M','U') and len(evidence)>30
        item=bank[sid]
        for im in item['images']:assert hashlib.sha256((root/im['path']).read_bytes()).hexdigest()==im['sha256']
        rows.append(dict(item,completion=completion,quality=quality,evidence=evidence,
            manifest=str(path.relative_to(root)),manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            reviewer='Codex direct native-pixel region/whole-image AI review; not human',
            at=datetime.datetime.now(datetime.timezone.utc).isoformat(),supplements_raw_primary_sheet=True,training_label_admitted=False))
        done.add(sid)
    dest.write_text(old+''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));print('Native-pixel supplements:',len(rows))

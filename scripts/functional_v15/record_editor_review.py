"""Record actually viewed anonymous editor-control images without selecting by metrics."""
import datetime, hashlib, json
from .native_labels import ROOT, EXP


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def record(key, label, stage, completion, quality, evidence):
    assert label in ('A','B') and stage in ('overview','native')
    assert completion in 'CPFU' and quality in 'AmMU' and len(evidence)>30
    manifest=EXP/'editor-only-review/manifest-readable.json'
    item=next(i for i in json.loads(manifest.read_text())['items'] if i['key']==key)
    assert digest(ROOT/item['sheet'])==item['sheet_sha256']
    assets=[i for i in item['native'] if i['label'] in ('INPUT',label)]
    assert len(assets)==2 and all(digest(ROOT/i['path'])==i['sha256'] for i in assets)
    dest=EXP/'editor-control-observations.jsonl';old=dest.read_text() if dest.exists() else ''
    previous=[json.loads(l) for l in old.splitlines()]
    assert (key,label,stage) not in {(i['key'],i['label'],i['stage']) for i in previous}
    row=dict(key=key,case_id=item['case_id'],split=item['split'],label=label,stage=stage,
        completion=completion,quality=quality,evidence=evidence,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        manifest_sha256=digest(manifest),sheet=item['sheet'],sheet_sha256=item['sheet_sha256'],assets=assets,
        requests=item['unchanged_case_requests'],language_rows=item['language_rows'],
        reviewer='Codex direct anonymous-image AI review; not human or user approval',
        scope='Editor-only fixed-parameter comparison; zero new model forwards or learning evidence.')
    temp=dest.with_suffix('.tmp');temp.write_text(old+json.dumps(row,ensure_ascii=False)+'\n');temp.replace(dest)
    print('Editor-control observation preserved',key,label,stage,completion,quality)

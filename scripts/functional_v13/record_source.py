"""Store directly viewed source notes and unadmitted parameter proposals."""
import json,datetime,hashlib
from pathlib import Path


def record(sheet_name, notes):
    root=Path(__file__).resolve().parents[2];exp=root/'experiments/functional-v13-20261106'
    m=json.loads((exp/'source-review-extended/manifest.json').read_text())
    s=next(s for s in m['sheets'] if Path(s['path']).name==sheet_name)
    assert hashlib.sha256((root/s['path']).read_bytes()).hexdigest()==s['sha256']
    assert len(notes)==len(s['ids'])
    bank={c['id']:c for c in m['candidates']};dest=exp/'source-observations.jsonl'
    old=dest.read_text() if dest.exists() else '';prior={json.loads(l)['id'] for l in old.splitlines()}
    assert not prior.intersection(s['ids'])
    rows=[]
    for sid,n in zip(s['ids'],notes):
        assert isinstance(n['retain'],bool) and len(n['evidence'])>20
        if n['retain']:
            assert n['primary_origin'] in ('original_GIER_English_AI_Chinese','AI_new_global_request')
            assert n['zh'] and n['en']
            assert n.get('action','edit') in ('edit','keep','decline')
            if n.get('action','edit')=='edit':assert n['parameters']
            if n.get('action')=='decline':assert n['reason'] in ('unsupported_tools','unreliable_quality')
            if n['primary_origin'].startswith('original'):assert n['en']==bank[sid]['instruction']
        rows.append(dict(n,id=sid,domain=bank[sid]['domain'],source_sha256=bank[sid]['source_sha256'],
            original_instruction=bank[sid]['instruction'],group_id=bank[sid]['group_id'],sheet=s['path'],sheet_sha256=s['sha256'],
            reviewer='Codex direct source/request AI observation, not human',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            parameter_status='unadmitted_proposal_must_execute_and_visually_verify',training_label_admitted=False))
    dest.write_text(old+''.join(json.dumps(n,ensure_ascii=False)+'\n' for n in rows))
    print('Stored',len(rows),'source notes; no labels admitted')

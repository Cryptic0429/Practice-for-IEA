"""Record direct candidate-image observations, separate from training admission."""
import json,hashlib,datetime
from pathlib import Path


def record(notes, bundle='primary-candidates', observations='primary-observations.jsonl'):
    root=Path(__file__).resolve().parents[2];exp=root/'experiments/functional-v13-20261106'
    manifest=exp/bundle/'manifest.json';m=json.loads(manifest.read_text());bank={r['id']:r for r in m['items']}
    dest=exp/observations;old=dest.read_text() if dest.exists() else ''
    done={json.loads(l)['id'] for l in old.splitlines()};rows=[]
    for sid,choice,completion,quality,evidence,alternatives in notes:
        assert sid not in done and completion in ('C','P','F','U') and quality in ('A','m','M','U')
        item=bank[sid];assert hashlib.sha256((root/item['sheet']).read_bytes()).hexdigest()==item['sheet_sha256']
        assert choice is None or choice in {r['candidate'] for r in item['results']}
        assert len(evidence)>30 and len(alternatives)>20
        rows.append(dict(id=sid,chosen_candidate=choice,completion=completion,quality=quality,evidence=evidence,
            alternatives=alternatives,sheet=item['sheet'],sheet_sha256=item['sheet_sha256'],
            source_input_sha256=item['input_sha256'],chosen_image_sha256=next((r['image_sha256'] for r in item['results'] if r['candidate']==choice),None),
            reviewer='Codex direct candidate-image visual AI review; not human',
            at=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_label_admitted=False,
            original_size_followup_required=quality=='U'))
        done.add(sid)
    dest.write_text(old+''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));print('Stored',len(rows),'primary candidate observations; no training admission')

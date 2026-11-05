"""Preserve disagreements on byte-identical historical regression results as uncertain."""
import argparse,json
from functional_v1.common import ROOT,read,write,append,sha,stamp
from .source_bank import EXP

def main(name):
    dest=EXP/'review/old-native/codex-observations.jsonl'
    current={r['item_id']:r for r in map(json.loads,dest.read_text().splitlines())}
    prior={};sources={}
    for v in (2,3,4,5):
        for path in (ROOT/'experiments').glob(f'functional-v{v}-*/review/natural-use/codex-observations.jsonl'):
            sources[str(path.relative_to(ROOT))]=sha(path)
            old={r['item_id']:r for r in map(json.loads,path.read_text().splitlines())}
            for key,row in old.items():prior.setdefault(key,[]).append((path,row))
    result=[]
    for key,fresh in current.items():
        if fresh.get('adjudication')=='historical_exact_render_disagreement':continue
        old=prior.get(key,[])
        if not old:continue
        assert all(row['image_sha256']==fresh['image_sha256'] for _,row in old)
        fields=[f for f in ('completion','quality') if any(row[f]!=fresh[f] for _,row in old)]
        if not fields:continue
        evidence=dict(item_id=key,case_id=fresh['case_id'],fresh={f:fresh[f] for f in ('completion','quality','evidence')},
            prior=[dict(file=str(p.relative_to(ROOT)),**{f:r[f] for f in ('completion','quality','evidence')}) for p,r in old],disagreed_fields=fields)
        result.append(evidence)
        revised=dict(fresh)
        for f in fields:revised[f]='uncertain'
        revised.update(adjudication='historical_exact_render_disagreement',previous_current_assessment_sha256=__import__('hashlib').sha256(json.dumps(fresh,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
            evidence='Repeated direct AI reviews of this exact semantic request/render disagree on '+', '.join(fields)+'. Both original observations are retained in the consistency audit; disputed fields remain uncertain and are not counted as qualified. No user adjudication.',observed_at=stamp(),consistency_audit=name)
        append(dest,revised)
    path=EXP/'review/old-native'/f'consistency-{name}.json';assert not path.exists()
    write(path,dict(at=stamp(),historical_source_hashes=sources,disagreements=result,
        scope='Same semantic/output hash only. Older reviews include inherited records, not independent reviewers. Original files and first fresh observations stay immutable.',policy='Disputed fields uncertain; never silently reuse a favourable opinion. Different outputs still require actual direct viewing.'))
    print(json.dumps(dict(new_uncertain_unique_outputs=len(result),audit=str(path.relative_to(ROOT)))))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);a=ap.parse_args();main(a.name)

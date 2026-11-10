"""Carry exact old controlled-image judgments, preserving every disagreement."""
import copy
import json
from collections import defaultdict
from functional_v1.common import ROOT,write,sha,stamp
from .runtime import EXP


def main():
    dest=EXP/'review/development/codex-observations.jsonl';assert not dest.exists()
    sources=[];groups=defaultdict(list)
    for version in (1,2,3,4):
        for path in (ROOT/'experiments').glob(f'functional-v{version}-*/review/development/codex-observations.jsonl'):
            sources.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
            # Last note in each old file may be its explicit uncertainty adjudication.
            bank={r['item_id']:r for r in map(json.loads,path.read_text().splitlines())}
            for key,n in bank.items():
                assert sha(ROOT/n['sheet'])==n['sheet_sha256']
                groups[key].append((path,n))
    rows=[];disputes=[]
    for key,values in groups.items():
        assert len({n['image_sha256'] for _,n in values})==1
        n=copy.deepcopy(values[-1][1]);fields=[f for f in ('completion','quality') if len({v[f] for _,v in values})>1]
        if fields:
            n.update({f:'uncertain' for f in fields})
            n['evidence']='Prior direct AI observations of the exact same render disagree on '+', '.join(fields)+'. All old notes are preserved; disagreement remains uncertain, without human adjudication.'
            disputes.append(dict(item_id=key,fields=fields))
        n['prior_final_observation_source']=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),assessment=v) for p,v in values]
        n['reuse_policy']='Exact input/request/criteria/output key only; inherited AI evidence, never a new independent image review or user approval.'
        rows.append(n)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(''.join(json.dumps(n,ensure_ascii=False)+'\n' for n in rows))
    (dest.parent/'codex-full-resolution-observations.jsonl').write_text('')
    write(EXP/'controlled-review-reuse.json',dict(at=stamp(),sources=sources,inherited_observations=len(rows),disputes=disputes,
        exact_match_required_for_use=True,new_image_reviews=0,acceptance_opened=False))
    print('Inherited controlled observations',len(rows),'conflicts kept U',len(disputes))


if __name__=='__main__':main()

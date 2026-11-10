"""Reuse prior final AI grades only for exact input/request/criteria/render keys."""
import json
from functional_v1.common import ROOT,read,write,sha,stamp
from .runtime import EXP


def main():
    receipt=EXP/'review-reuse.json';assert not receipt.exists();records=[]
    for split in ('native-development','old-native','development'):
        all_notes={};sources={}
        for path in (ROOT/'experiments/functional-v13-20261106/evaluation').glob(f'*/{split}/*/direct-review.json'):
            sources[str(path.relative_to(ROOT))]=sha(path)
            for row in read(path):
                key=row.get('item_id');note=row['assessment']
                if not key:continue
                assert note['item_id']==key and sha(ROOT/note['sheet'])==note['sheet_sha256']
                all_notes.setdefault(key,[]).append((path,note))
        notes=[]
        for key,items in all_notes.items():
            note=dict(items[0][1]);assert all(n['image_sha256']==note['image_sha256'] for _,n in items)
            conflicts=[field for field in ('completion','quality') if len({n[field] for _,n in items})>1]
            for field in conflicts:note[field]='uncertain'
            if conflicts:note['evidence']='Prior final AI assessments of this exact input/request/render disagree on '+','.join(conflicts)+'; uncertainty retained without a new vote.'
            note['prior_final_sources']=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p,_ in items]
            note['reuse_policy']='Exact item key only; prior final AI evidence and unresolved uncertainty, not new viewing.'
            notes.append(note)
        dest=EXP/'review'/split/'codex-observations.jsonl';assert not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(''.join(json.dumps(n,ensure_ascii=False)+'\n' for n in notes));(dest.parent/'codex-full-resolution-observations.jsonl').write_text('')
        records.append(dict(split=split,sources=sources,unique_observations=len(notes),destination=str(dest.relative_to(ROOT)),initial_sha256=sha(dest)))
    write(receipt,dict(at=stamp(),records=records,actual_new_visual_reviews=0,policy='Exact renders can inherit final AI assessments. Every unmatched output still requires anonymous direct review; historical disagreements remain uncertain.'))
    print('Seeded exact prior final observations, no new quality claim.')


if __name__=='__main__':main()

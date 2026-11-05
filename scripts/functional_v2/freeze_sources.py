"""Freeze only directly classified source photos; audit inherited-training leakage too."""
import collections,json
from functional_v1.prepare import fingerprint,near
from functional_v1.editor import load_rgb
from .common import ROOT,DATA,EXP,read,write,sha,stamp

def main():
    out=DATA/'source-splits.json';assert not out.exists()
    manifest=read(EXP/'source-review/manifest.json');bank={r['id']:r for r in manifest['candidates']}
    notes=[json.loads(l) for l in (EXP/'source-review/observations.jsonl').read_text().splitlines()]
    chosen=[{**bank[n['id']],'domain':n['domain'],'classification_evidence':n['evidence']} for n in notes if n['domain']!='reject']
    assert len(chosen)==100 and len({r['id'] for r in chosen})==100
    features={r['id']:fingerprint(load_rgb(r['source_image'])) for r in chosen}
    clashes=[]
    for i,a in enumerate(chosen):
        assert sha(a['source_image'])==a['source_sha256']
        for b in chosen[:i]:
            if a['group_id']==b['group_id'] or near(features[a['id']],features[b['id']]):clashes.append({'a':a['id'],'b':b['id'],'reason':'same_or_near_selected_sources'})
    historical=read(ROOT/'data/training/quality-v2-20261103/train3k.json')
    historic={r['group_id']:r for r in historical}
    oldheld={r['group_id']:r for s in ('development','acceptance') for r in read(ROOT/'data/training/functional-v1-20261104'/f'{s}-cases.json')}
    for i,(gid,r) in enumerate(historic.items()):
        f=fingerprint(load_rgb(r['images'][0]))
        for a in chosen:
            if a['split']=='train':continue
            if gid==a['group_id'] or near(f,features[a['id']]):clashes.append({'a':a['id'],'b':gid,'reason':'near_inherited_train3k'})
    for gid,r in oldheld.items():
        f=fingerprint(load_rgb(ROOT/r['image']))
        for a in chosen:
            if a['split']!='train':continue
            if gid==a['group_id'] or near(f,features[a['id']]):clashes.append({'a':a['id'],'b':gid,'reason':'new_train_near_prior_holdout'})
    audit={'at':stamp(),'selected_sources':len(chosen),'inherited_train_groups_checked':len(historic),'prior_holdout_groups_checked':len(oldheld),
        'method':'original group IDs plus pHash Hamming <=6 and grayscale correlation >.985 or MAE <.01; cannot guarantee detection of all crops/related shoots',
        'counts':{s:dict(collections.Counter(r['domain'] for r in chosen if r['split']==s)) for s in ('train','development','acceptance')},'collisions':clashes}
    write(EXP/'planning/source-isolation-audit.json',audit);print(json.dumps(audit,ensure_ascii=False))
    if clashes:raise RuntimeError('Replace colliding source photos before freezing')
    write(out,{'status':'frozen_before_round2_training','at':stamp(),'source_bank_sha256':sha(EXP/'source-review/manifest.json'),
        'observations_sha256':sha(EXP/'source-review/observations.jsonl'),'sources':chosen,'isolation_audit_sha256':sha(EXP/'planning/source-isolation-audit.json'),
        'history':'Original validation/test pools have historical evaluation exposure; fresh to this round, not newly collected independent user photos.'})
if __name__=='__main__':main()

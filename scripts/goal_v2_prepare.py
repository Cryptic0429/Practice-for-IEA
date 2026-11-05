"""Freeze new holdout and an isolated development campaign; CPU only."""
import collections, hashlib, json
from pathlib import Path
ROOT = Path('/mnt/sdc/zhc/IEA')
OUT = ROOT/'data/goal-v2-20261103'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not OUT.exists(), 'Never replace a frozen holdout'
    old=ROOT/'data/stage1/splits/image-edit-3k-v1'
    groups=read(old/'group-assignments.json')
    path_group={p:g['group_id'] for g in groups for p in g['source_paths']}
    ready=ROOT/'data/stage1/ready/GIER-Edit'
    allrows=read(ready/'sft_train.json')+read(ready/'sft_test.json')
    idgroup={r['id']:path_group[r['extra_info']['origin_image_path']] for r in allrows}
    used=set(); evidence=[]
    def scan(v):
        if isinstance(v,dict):
            for x in v.values(): yield from scan(x)
        elif isinstance(v,list):
            for x in v: yield from scan(x)
        elif isinstance(v,str): yield v
    sources=set((ROOT/'data/stage1/splits').rglob('*train*.json'))
    sources.update((ROOT/'logs').rglob('*.json'))
    sources.update((ROOT/'outputs').rglob('predictions.jsonl'))
    sources.update((ROOT/'outputs').rglob('candidates.jsonl'))
    sources.update((ROOT/'data/stage2').rglob('test*.json'))
    sources.update([old/'validation.json', old/'test.json'])
    for p in sorted(sources):
        if p.name in {'stage1-data-readiness.json','frozen-sequence-audit.json'}: continue
        values=[json.loads(l) for l in p.read_text().splitlines()] if p.suffix=='.jsonl' else [read(p)]
        strings=set(scan(values)); found=(strings & set(path_group.values()))
        found|={path_group[s] for s in strings if s in path_group}
        found|={idgroup[s] for s in strings if s in idgroup}
        used|=found; evidence.append({'path':str(p),'sha256':sha(p),'groups':sorted(found)})
    previous=read(ROOT/'data/stage2/stage2-A-edit-grpo-v1/manifest.json')
    used.update(previous['historically_excluded_candidates'])
    pool=collections.defaultdict(list)
    for r in allrows:
        g=path_group[r['extra_info']['origin_image_path']]
        if g not in used: pool[g].append(dict(r,group_id=g))
    holdout=[sorted(pool[g],key=lambda r:r['id'])[0] for g in sorted(pool)]
    assert len(holdout)>=30, len(holdout)
    train=read(ROOT/'data/stage2/stage2-A-edit-grpo-v1/train.json')
    validation=read(old/'validation.json')
    assert not set(pool)&{r['group_id'] for r in train+validation}
    OUT.mkdir(parents=True)
    files={}
    for name,rows in [('train',train),('validation',validation),('test_sealed',holdout)]:
        p=OUT/f'{name}.json';p.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
        files[name]={'path':str(p),'sha256':sha(p),'rows':len(rows),'groups':len({r['group_id'] for r in rows})}
    result={'status':'frozen','files':files,'usage_evidence':evidence,
      'limitations':['Record-based exclusion, not upstream-pretraining or perceptual-near-duplicate certification.',
                    'Holdout never used for iteration; limited group count may give wide intervals.'],
      'acceptance':{'relative_paired_mean_l1_improvement':0.03,'multi_seed_minimum':3,
                    'baseline':'minimum full-validation L1 among A/B/SFT/GRPO in identical runtime, frozen before new tests',
                    'coverage':'no lower than baseline on test; report failure-fallback separately',
                    'nonoperation':'report identical pixels and empty/zero parameters; no increase used as success',
                    'instruction':'predefined explicit-direction subgroup plus blinded human review if available; AI review disclosed'}}
    (OUT/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'frozen','files':files},indent=2))
if __name__=='__main__': main()

"""Freeze the Image-Edit pilot by original-image identity, without evaluating models."""
import collections
import copy
import hashlib
import json
import random
import urllib.parse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path('/mnt/sdc/zhc/IEA')
OUT = ROOT/'data/stage1/splits/image-edit-3k-v1'
SEED = 20261031


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False)+'\n'
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f'Refusing to change an already-frozen file: {path}')
    path.write_text(text)


def main():
    if (OUT/'manifest.json').exists():
        manifest = json.loads((OUT/'manifest.json').read_text())
        for item in manifest['files'].values():
            assert sha(Path(item['path'])) == item['sha256']
        print('EXISTING_FROZEN_SPLIT_VERIFIED')
        return
    sources = {s: ROOT/f'data/stage1/ready/GIER-Edit/sft_{s}.json' for s in ['train','test']}
    datasets = {s: json.loads(p.read_text()) for s,p in sources.items()}
    images = json.loads((ROOT/'data/stage1/image-manifest.json').read_text())
    metadata = json.loads((ROOT/'repo/datasets/GIER/GIER.json').read_text())
    urls = {x['input']: x.get('input_url') for x in metadata}
    origins = sorted({r['extra_info']['origin_image_path'] for ds in datasets.values() for r in ds})
    parent = {p:p for p in origins}
    def find(p):
        while parent[p]!=p:
            parent[p]=parent[parent[p]]
            p=parent[p]
        return p
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    seen = {}
    for path in origins:
        keys = [('pixel',images[path]['pixel_sha256'])]
        url=urls.get(Path(path).name)
        if url:
            parsed=urllib.parse.urlsplit(url)
            # HTTP/HTTPS are the same source identity. Preserve path and query.
            keys.append(('source_url',parsed.netloc.lower()+parsed.path+'?'+parsed.query))
        for key in keys:
            if key in seen: union(path,seen[key])
            else: seen[key]=path
    members=collections.defaultdict(list)
    for path in origins: members[find(path)].append(path)
    group_ids={p:hashlib.sha256('\n'.join(sorted(members[find(p)])).encode()).hexdigest() for p in origins}
    grouped={s:collections.defaultdict(list) for s in datasets}
    for split,ds in datasets.items():
        for item in ds:
            row=copy.deepcopy(item)
            row['group_id']=group_ids[row['extra_info']['origin_image_path']]
            grouped[split][row['group_id']].append(row)
    test_reserved=set(grouped['test'])
    additional_overlap=sorted(set(grouped['train']) & test_reserved)
    available=sorted(set(grouped['train'])-test_reserved)
    rng=random.Random(SEED)
    val_groups=set(rng.sample(available,200))
    test_groups=set(rng.sample(sorted(test_reserved),200))
    val=[rng.choice(sorted(grouped['train'][g],key=lambda r:r['id'])) for g in sorted(val_groups)]
    test=[rng.choice(sorted(grouped['test'][g],key=lambda r:r['id'])) for g in sorted(test_groups)]
    pool=[r for g in sorted(set(available)-val_groups) for r in grouped['train'][g]]
    # Preserve instruction-type proportions of the eligible pool; no rare-tool upsampling for A.
    by_type=collections.defaultdict(list)
    for row in pool: by_type[row['extra_info']['inst_type']].append(row)
    target={k:3000*len(v)/len(pool) for k,v in by_type.items()}
    quotas={k:int(v) for k,v in target.items()}
    for k in sorted(target,key=lambda k:(-(target[k]-quotas[k]),k))[:3000-sum(quotas.values())]: quotas[k]+=1
    train=[]
    for kind in sorted(by_type): train.extend(rng.sample(sorted(by_type[kind],key=lambda r:r['id']),quotas[kind]))
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    probe=random.Random(SEED+1).sample(train,256)
    splits={'train':train,'validation':val,'test':test,'probe_train':probe}
    def stats(rows):
        return dict(rows=len(rows),unique_source_paths=len({r['extra_info']['origin_image_path'] for r in rows}),
                    unique_groups=len({r['group_id'] for r in rows}),
                    instruction_types=dict(collections.Counter(r['extra_info']['inst_type'] for r in rows)),
                    nonzero_tool_counts=dict(collections.Counter(k for r in rows for k,v in r['extra_info']['label'].items() if v!=0)))
    group_sets={s:{r['group_id'] for r in rows} for s,rows in splits.items()}
    for a,b in [('train','validation'),('train','test'),('validation','test')]: assert not group_sets[a]&group_sets[b]
    assert len(train)==3000 and len(val)==len(test)==200
    assert len({r['id'] for r in train})==3000
    pool_tools={k for r in pool for k,v in r['extra_info']['label'].items() if v!=0}
    assert set(stats(train)['nonzero_tool_counts']) == pool_tools
    files={}
    for split,rows in splits.items():
        path=OUT/f'{split}.json'
        dump(path,rows)
        files[split]=dict(path=str(path),sha256=sha(path),**stats(rows))
        dump(OUT/f'{split}-ids.json',[dict(id=r['id'],source_id=r['source_id'],group_id=r['group_id']) for r in rows])
    assignments=[]
    for representative,paths in sorted(members.items()):
        group=group_ids[representative]
        partition='test_reserved' if group in test_reserved else 'validation_reserved' if group in val_groups else 'training_pool'
        assignments.append(dict(group_id=group,partition=partition,source_paths=sorted(paths),
                                pixel_hashes=sorted({images[p]['pixel_sha256'] for p in paths})))
    dump(OUT/'group-assignments.json',assignments)
    manifest=dict(created_at=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),seed=SEED,
                  task='Image-Edit',version='image-edit-3k-v1',files=files,
                  source_files={s:dict(path=str(p),sha256=sha(p)) for s,p in sources.items()},
                  eligible_training_pool_rows=len(pool),eligible_training_pool_groups=len(set(available)-val_groups),
                  all_official_test_groups_reserved=len(test_reserved),validation_groups_reserved=len(val_groups),
                  additional_source_url_overlap_groups=additional_overlap,
                  group_assignment_sha256=sha(OUT/'group-assignments.json'),
                  cross_split_source_group_overlap=0,
                  sampling='Seeded 3000-row sample stratified by instruction type, proportional to eligible pool; validation/test use one uniform random row per sampled group.',
                  group_definition='Connected components of original image path, identical decoded RGB pixels, and identical source URL after scheme normalization.',
                  scope_notes=['All rows and any future Summary rows belonging to validation/test groups are excluded from training.',
                               'Test subset is fixed before any pilot model evaluation; it is not the complete paper benchmark.',
                               'Perceptual near-duplicates with different source URLs and pixels are not certified absent.'])
    dump(OUT/'manifest.json',manifest)
    registry_path=ROOT/'data/dataset_info.json'
    registry=json.loads(registry_path.read_text())
    for split in splits:
        entry=copy.deepcopy(registry['gier_sft_distill'])
        entry['file_name']=files[split]['path']
        registry['iea_edit_3k_'+split]=entry
    registry_path.write_text(json.dumps(registry,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()

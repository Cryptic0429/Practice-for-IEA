"""Reuse only the frozen original3k training pool, excluding every project holdout."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path('/mnt/sdc/zhc/IEA');DATA=ROOT/'data/goal-v2-20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=ROOT/'data/stage1/splits/image-edit-3k-v1/train.json'
    assert sha(source)=='439ebf726d7e01c913a9e85c0866240268edb3b5919bb583b4823270fe361104'
    rows=json.loads(source.read_text());groups={x['group_id'] for x in rows};ids={x['id'] for x in rows};assert len(rows)==len(ids)==3000
    paths={'validation':DATA/'validation.json','sealed71':DATA/'test_sealed.json',
           'old_test200':ROOT/'data/stage1/splits/image-edit-3k-v1/test.json',
           'stage2_new100':ROOT/'data/stage2/stage2-A-edit-grpo-v1/test_new.json'}
    checks={}
    for name,p in paths.items():
        other=json.loads(p.read_text());overlap=groups&{x['group_id'] for x in other};assert not overlap,name
        checks[name]={'rows':len(other),'source_sha256':sha(p),'overlap_groups':len(overlap)}
    assert {x['id'] for x in json.loads((DATA/'train.json').read_text())}.issubset(ids)
    dest=DATA/'train3k.json';assert not dest.exists();shutil.copyfile(source,dest)
    manifest={'status':'frozen_train3k_only','files':{'train3k':{'path':str(dest),'sha256':sha(dest),'rows':3000,'groups':len(groups),'source_path':str(source),'source_sha256':sha(source)}},
              'overlap_checks':checks,'test_images_or_references_scored':False,'purpose':'Broader coverage control using original A training pool; no expansion into heldouts'}
    (DATA/'training-variants-3k.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()

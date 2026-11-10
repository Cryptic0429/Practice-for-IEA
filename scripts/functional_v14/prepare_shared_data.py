"""Reuse identical admitted R13 records/cache without inventing new labels or photos."""
import shutil
from collections import Counter
from functional_v1.common import ROOT,read,write,sha,stamp
from .runtime import EXP,DATA


def main():
    old=ROOT/'data/training/functional-v13-20261106';prior=ROOT/'experiments/functional-v13-20261106'
    assert not read(prior/'development-selection.json')['qualified']
    assert not (DATA/'admission.json').exists()
    DATA.mkdir(parents=True,exist_ok=True)
    for name in ('cases.json','train-accepted.json'):
        shutil.copy2(old/name,DATA/name);assert sha(old/name)==sha(DATA/name)
    cases=read(DATA/'cases.json');rows=read(DATA/'train-accepted.json')
    assert len(cases)==925 and len(rows)==1850 and len({c['group_id'] for c in cases})==240
    assert Counter(c['group_id'] for c in cases)==Counter(c['group_id'] for c in read(old/'cases.json'))
    assert all(Counter(r['language'] for r in rows if r['case_id']==c['id'])==dict(zh=1,en=1) for c in cases)
    admission=read(old/'admission.json')
    admission.update(at=stamp(),round14_new_labels=0,round14_new_original_photos=0,round14_direct_new_visual_reviews=0,
        inherited_R13_cases=925,inherited_admission=str((old/'admission.json').relative_to(ROOT)),inherited_admission_sha256=sha(old/'admission.json'),
        status='Exact925_R13_cases_inherited_with_existing_AI_evidence_not_new_independent_review')
    write(DATA/'admission.json',admission)
    cache=read(prior/'batch-cache.json');assert sha(ROOT/cache['cache_path'])==cache['cache_sha256']
    cache.update(round14_reuse=True,source=str((prior/'batch-cache.json').relative_to(ROOT)),source_sha256=sha(prior/'batch-cache.json'))
    write(EXP/'batch-cache.json',cache)
    evidence={}
    for name in ('target-replay.json','input-audit.json','editor-invariant-tests.json','outcome-contract-check.json'):
        path=prior/name;assert read(path)['status']=='passed';evidence[name]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),assessment='Exact data and same input/parser/editor implementation; prior successful check inherited, not newly claimed visual review.')
    write(EXP/'shared-data-audit.json',dict(at=stamp(),status='passed',semantic_cases=925,language_records=1850,original_photo_groups=240,
        datasets_byte_identical_to_R13=True,new_labels=0,new_images=0,prior_evidence=evidence,CPU_cache_path=cache['cache_path'],CPU_cache_sha256=cache['cache_sha256'],GPU_hours=0))
    print('Exact R13 data/cache reuse:925/1850/240; no new labels/photos or invented review.')


if __name__=='__main__':main()

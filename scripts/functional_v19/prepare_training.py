"""CPU-only exact positive-data reuse and current-image feature cache."""
import shutil
from collections import Counter
import numpy as np
from functional_v1.common import ROOT,read,write,sha,stamp
from .photometry import features,FEATURE_NAMES

EXP=ROOT/'experiments/functional-v19-20261107';DATA=ROOT/'data/training/functional-v19-20261107'

def main():
    assert not (DATA/'admission.json').exists()
    source=ROOT/'data/training/functional-v18-20261107';p=read(ROOT/'experiments/functional-v18-20261107/training-protocol.json')
    assert read(ROOT/'experiments/functional-v18-20261107/development-selection.json')['status']=='failed_practical_gate'
    assert read(EXP/'bridge-contract-check.json')['status']=='CPU_tiny_actual_Qwen_PEFT_checks_passed'
    cases=read(source/'cases.json');rows=read(source/'train-accepted.json');pool=read(source/'training-pool.json')
    assert (len(cases),len(rows),len(pool))==(925,1850,2002)
    assert sha(source/'cases.json')==p['cases_sha256'] and sha(source/'train-accepted.json')==p['dataset_sha256']
    image_records={};case_features={}
    for c in cases:
        path=ROOT/c['image'];assert sha(path)==c['input_sha256'] and c['split']=='train'
        if c['image'] not in image_records:
            a=features(path);b=features(path);assert np.array_equal(a,b)
            image_records[c['image']]=dict(input_sha256=c['input_sha256'],features=a.tolist())
        case_features[c['id']]=dict(image=c['image'],**image_records[c['image']])
    pairs=read(ROOT/'experiments/functional-v18-20261107/preference-selection.json')
    # Source isolation follows exact unchanged R15/R18 photo groups and labels;
    # this cache never reads any development/acceptance image or target.
    DATA.mkdir(parents=True,exist_ok=False)
    for name in ('cases.json','train-accepted.json','training-pool.json'):
        shutil.copy2(source/name,DATA/name);assert sha(DATA/name)==sha(source/name)
    write(EXP/'input-features.json',dict(at=stamp(),feature_names=FEATURE_NAMES,cases=case_features,
        unique_image_paths=len(image_records),deterministic_recompute_exact=True,input_only=True,targets_or_instructions_read_by_feature_function=False,
        feature_source_sha256=sha(ROOT/'scripts/functional_v19/photometry.py'),GPU_calls=0))
    write(DATA/'admission.json',dict(at=stamp(),status='exact_unchanged_visual_labels_inherited_new_input_RGB_branch_only',
        semantic_cases=925,language_records=1850,source_groups=240,domains=dict(Counter({c['group_id']:c['domain'] for c in cases}.values())),
        actions=dict(Counter(c['action'] for c in cases)),original_English=154,AI_requests=771,new_photos=0,new_requests=0,new_positive_labels=0,
        repeated_positive_presentations_per_epoch=2002,extra_preference_positive_semantics=19,
        preference_ranking_used=False,negative_images_used=False,new_target_visual_reviews=0,
        dataset_sha256=sha(DATA/'train-accepted.json'),cases_sha256=sha(DATA/'cases.json'),pool_sha256=sha(DATA/'training-pool.json'),
        input_features_sha256=sha(EXP/'input-features.json'),source_admission_sha256=sha(source/'admission.json'),
        label_and_isolation_evidence='Exact R18/R15 labels and240 TRAIN groups unchanged; inherited audit and full-pool fingerprints, not a new independent full scan.',
        base_cache_path=p['base_cache_path'],base_cache_sha256=p['base_cache_sha256'],independent_user_feedback=0,
        comparison='Exact R18 matched-SFT positive pool/order/budget; this route additionally trains a16→128→3584 input-RGB bridge. Capacity and feature-path change together; not a separate trained constant-feature control.'))
    print('CPU prepared925 unchanged semantics/1850language/2002presentations; input RGB features',len(image_records),'unique image paths; GPU0.')

if __name__=='__main__':main()

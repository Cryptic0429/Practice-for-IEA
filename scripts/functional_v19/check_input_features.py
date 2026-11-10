"""CPU feature provenance/polarity checks, without quality labels or model scoring."""
from collections import defaultdict,Counter
import numpy as np
from functional_v1.common import ROOT,read,write,sha,stamp
from .photometry import features,FEATURE_NAMES

EXP=ROOT/'experiments/functional-v19-20261107';DATA=ROOT/'data/training/functional-v19-20261107'
def main():
    frozen=read(EXP/'input-features.json');cases=read(DATA/'cases.json');cache=frozen['cases']
    assert frozen['feature_names']==list(FEATURE_NAMES)
    assert sha(ROOT/'scripts/functional_v19/photometry.py')==frozen['feature_source_sha256']
    identities=defaultdict(list)
    for c in cases:identities[c['input_sha256']].append(c['id'])
    assert all(len({tuple(cache[cid]['features']) for cid in ids})==1 for ids in identities.values())
    results=[]
    for mild in cases:
        if not mild['id'].endswith('-mild'):continue
        strong_id=mild['id'][:-5]+'-strong'
        if strong_id not in cache:continue
        if '-restore-exposure-' in mild['id']:index=0;kind='exposure'
        elif '-restore-contrast-' in mild['id']:index=1;kind='contrast'
        elif '-restore-saturation-' in mild['id']:index=14;kind='saturation'
        else:continue
        a=cache[mild['id']]['features'][index];b=cache[strong_id]['features'][index]
        assert b<a,(mild['id'],FEATURE_NAMES[index],a,b)
        results.append(dict(mild=mild['id'],strong=strong_id,feature=FEATURE_NAMES[index],mild_value=a,strong_value=b,expected_direction=True))
    assert results
    write(EXP/'feature-provenance-check.json',dict(at=stamp(),status='CPU_input_only_feature_checks_passed',
        semantic_cases=len(cases),unique_input_byte_hashes=len(identities),identical_input_same_features=True,
        checked_synthetic_mild_strong_pairs=len(results),records=results,
        semantic_domains=dict(Counter(c['domain'] for c in cases)),
        inference_features_from_current_RGB_only=True,labels_targets_requests_not_feature_arguments=True,
        feature_source_sha256=sha(ROOT/'scripts/functional_v19/photometry.py'),GPU_calls=0,
        limitation='These input statistics have the expected direction on synthetic conditions; no editing quality, model correctness or unseen-image performance is inferred.'))
    print('Feature-only CPU checks:',len(identities),'distinct input bytes;',len(results),'synthetic direction pairs; no quality score.')

if __name__=='__main__':main()

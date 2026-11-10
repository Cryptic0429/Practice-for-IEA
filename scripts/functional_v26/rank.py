"""CPU same-source rank feasibility on explicit edited alternatives."""
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression

from functional_v1.common import ROOT,read,sha,stamp,write
from functional_v25.features import INPUT_NAMES,PAIR_NAMES,features
from .stage import DATA
from .shortlist import EXP

FIXED = {'fixed_shadows38':{'shadows':38},'fixed_shadows77':{'shadows':77},
         'fixed_exposure60':{'exposure':60}}
DIMENSIONS = {'parameters':2,'parameters_and_pixels':2+len(PAIR_NAMES)-len(INPUT_NAMES)}


def vector(case,variant):
    params = variant['decision']['parameters']
    inp,out = ROOT/case['image'],ROOT/variant['path']
    if sha(inp) != case['image_sha256'] or sha(out) != variant['sha256']:
        raise ValueError('Image evidence mismatch')
    pixels = features(inp,out)[len(INPUT_NAMES):]
    return np.array([params.get('exposure',0)/100,params.get('shadows',0)/100,*pixels])


def fit(pairs,vectors,k):
    differences = np.stack([vectors[p['case_id']][p['chosen']['id']][:k] -
                            vectors[p['case_id']][p['rejected']['id']][:k] for p in pairs])
    x = np.concatenate((differences,-differences))
    y = np.repeat([1,0],len(pairs))
    scale = np.maximum(x.std(axis=0),1e-8)
    group_counts = Counter(p['group_id'] for p in pairs)
    weights = [1/group_counts[p['group_id']] for p in pairs]*2
    model = LogisticRegression(C=.3,solver='liblinear',fit_intercept=False,max_iter=1000,random_state=20261107)
    return model.fit(x/scale,y,sample_weight=weights).coef_[0]/scale


def preference_score(chosen,rejected):
    return 'tie' if abs(chosen-rejected) <= 1e-9 else ('correct' if chosen>rejected else 'wrong')


def main():
    receipt = read(DATA/'receipt.json')
    cases_path,pairs_path = DATA/'reviewed-candidates.json',DATA/'preference-candidates.json'
    if (receipt['cases_sha256'],receipt['pairs_sha256']) != (sha(cases_path),sha(pairs_path)):
        raise ValueError('Candidate data differs from the visual review receipt')
    cases,pairs = read(cases_path),read(pairs_path)
    if len({p['group_id'] for p in pairs}) < 6:
        raise ValueError('Fewer than six labeled training source groups')
    vectors = {c['id']:{v['id']:vector(c,v) for v in c['candidates']} for c in cases}
    predicted = []
    for case in cases:
        cid,gid = case['id'],case['group_id']
        train = [p for p in pairs if p['group_id'] != gid]
        held = [p for p in pairs if p['group_id'] == gid]
        result = dict(case_id=cid,group_id=gid,domain=case['domain'],
                      available_completed=sum(v['verdict']=='complete' for v in case['candidates']),
                      candidates=len(case['candidates']),scores={},selected={},pair_verdicts={})
        for name,params in FIXED.items():
            choice = next(v for v in case['candidates'] if v['decision']['parameters']==params)
            result['selected'][name] = dict(id=choice['id'],verdict=choice['verdict'])
        for name,k in DIMENSIONS.items():
            weights = fit(train,vectors,k)
            scores = {v:float(x[:k]@weights) for v,x in vectors[cid].items()}
            # max is stable. Explicitly prefer the mild original setting on a tie.
            ordered = sorted(case['candidates'],key=lambda v:v['decision']['parameters']!={'shadows':38})
            selected = max(ordered,key=lambda v:scores[v['id']])
            result['scores'][name] = scores
            result['selected'][name] = dict(id=selected['id'],verdict=selected['verdict'])
            result['pair_verdicts'][name] = [dict(pair_id=p['id'],
                verdict=preference_score(scores[p['chosen']['id']],scores[p['rejected']['id']])) for p in held]
        predicted.append(result)
    summary = {}
    for name in [*FIXED,*DIMENSIONS]:
        summary[name] = dict(selected=dict(Counter(r['selected'][name]['verdict'] for r in predicted)),
            with_completed_available=dict(Counter(r['selected'][name]['verdict'] for r in predicted if r['available_completed'])),
            held_pairs=dict(Counter(x['verdict'] for r in predicted for x in r['pair_verdicts'].get(name,[]))))
    r24_data = ROOT/'data/training/functional-v24-20261107/reviewed-candidates.json'
    r24_receipt = read(r24_data.parent/'receipt.json')
    if sha(r24_data)!=r24_receipt['candidates_sha256']:
        raise ValueError('R24 candidate evidence changed')
    probe = next(c for c in read(r24_data) if c['id']=='p227')
    if probe['group_id'] in {c['group_id'] for c in cases}:
        raise ValueError('Historical probe overlaps training source groups')
    pref = probe['preference_candidate']
    best = vector(probe,dict(path=probe['target'],sha256=probe['target_sha256'],decision=pref['chosen']))
    bad = vector(probe,dict(path=pref['rejected_result'],sha256=pref['rejected_result_sha256'],decision=pref['rejected']))
    probe_scores = {}
    weights = {}
    for name,k in DIMENSIONS.items():
        w = fit(pairs,vectors,k)
        weights[name] = w.tolist()
        chosen,rejected = float(best[:k]@w),float(bad[:k]@w)
        probe_scores[name] = dict(chosen=chosen,rejected=rejected,verdict=preference_score(chosen,rejected))
    write(EXP/'ranking-result.json',dict(at=stamp(),protocol_sha256=sha(EXP/'ranking-protocol.json'),
        cases_sha256=sha(cases_path),pairs_sha256=sha(pairs_path),historical_r24_source_sha256=sha(r24_data),
        summary=summary,cases=predicted,training_pairs=len(pairs),
        fit_pair_groups=len({p['group_id'] for p in pairs}),historical_p227=probe_scores,full_fit_weights=weights,
        GPU_hours=0,deployed=False,final_acceptance_opened=False))


if __name__ == '__main__':
    main()

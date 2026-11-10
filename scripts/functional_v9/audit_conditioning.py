"""CPU diagnosis of exact-request visual supervision, not a model quality score."""
from collections import defaultdict,Counter
import json
from functional_v1.common import ROOT,read,write,stamp,sha

EXP=ROOT/'experiments/functional-v9-20261106'
DATA=ROOT/'data/training/functional-v7-20261105'

def main():
    out=EXP/'request-conditioning-audit.json';assert not out.exists()
    groups=defaultdict(list)
    for c in read(DATA/'cases.json'):
        groups[c['instructions']['en'].strip()].append(c)
    repeated=[]
    for instruction,cases in groups.items():
        photos={c['group_id'] for c in cases}
        if len(photos)<2:continue
        decisions=Counter(json.dumps(c['decision'],sort_keys=True) for c in cases)
        repeated.append(dict(instruction_en=instruction,semantic_cases=len(cases),photo_groups=len(photos),
            distinct_decision_targets=len(decisions),largest_identical_target_cases=max(decisions.values()),
            actions=dict(Counter(c['decision']['action'] for c in cases)),case_ids=[c['id'] for c in cases]))
    quality=[c for c in read(DATA/'cases.json') if c['decision'].get('reason')=='unreliable_quality']
    exact_counterfactual=[r for r in repeated if r['actions'].get('edit') and r['actions'].get('decline')]
    write(out,dict(at=stamp(),training_cases_sha256=sha(DATA/'cases.json'),semantic_cases=462,
        exact_English_request_groups=len(groups),multi_photo_exact_request_groups=len(repeated),
        repeated_groups=repeated,unreliable_quality_training_cases=[dict(id=c['id'],group_id=c['group_id'],instructions=c['instructions']) for c in quality],
        exact_request_groups_with_edit_and_quality_refusal=exact_counterfactual,
        limitation='Exact-string diagnosis only, not semantic paraphrase equivalence or a practical model score. Different valid parameter targets still require visual quality inspection.'))
    print(dict(exact_request_groups=len(groups),multi_photo_groups=len(repeated),quality_cases=len(quality),same_request_edit_and_decline=len(exact_counterfactual)))

if __name__=='__main__':main()

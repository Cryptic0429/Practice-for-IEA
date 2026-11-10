"""Select unchanged reviewed anchors, with explicit inherited evidence.

No new targets are judged by this script. The old input, request and target
must match the frozen Round7 admission and actual replay before reuse.
"""
import copy
from collections import Counter
from functional_v1.common import ROOT,read,write,sha,stamp
from .prepare_counterfactuals import EXP

PREVIOUS=ROOT/'data/training/functional-v7-20261105'


def main():
    destination=EXP/'anchor-candidates.json';assert not destination.exists()
    assert sha(PREVIOUS/'cases.json')=='1948fbe5e40cdeae1d3e37fefedd4c87d9ebb7a2cf1eec3ff830229e8b98353a'
    assert sha(PREVIOUS/'train-accepted.json')=='fe3f8e6479c5d74ea3783cc4edb643f1a2fc5a79ba50e826df5a5786ec8e068c'
    candidates=read(PREVIOUS/'cases.json');admission=read(PREVIOUS/'admission.json')
    assert admission['semantic_cases']==462
    selected=[];used={c['group_id'] for c in read(EXP/'counterfactual-candidates/manifest.json')['cases']}
    # Reserve the scarcer unsupported controls before the flexible AI edit
    # and keep choices; a greedy keep-first pass can consume their groups.
    def controls(action,counts):
        for domain,count in counts.items():
            choices=[c for c in candidates if c['action']==action and c['domain']==domain and c['group_id'] not in used and
                (action=='keep' or c['decision'].get('reason')=='unsupported_tools')]
            assert len(choices)>=count,(action,domain,len(choices),count)
            for c in sorted(choices,key=lambda c:c['id'])[:count]:
                assert c['group_id'] not in used;used.add(c['group_id']);selected.append(copy.deepcopy(c))
    controls('decline',dict(portrait=2,landscape=3))
    for family in ('lighting','white_balance','saturation','contrast','brown_tone'):
        for domain in ('portrait','landscape'):
            choices=[c for c in candidates if c['family']==family and c['domain']==domain and
                c['action']=='edit' and c['group_id'] not in used]
            # Original English has no landscape white-balance cases in this
            # family. Preserve task/domain coverage and disclose AI anchors.
            choices.sort(key=lambda c:(not c['instruction_origin'].startswith('English untouched original GIER'),c['id']))
            taken=0
            for c in choices:
                if c['group_id'] in used:continue
                used.add(c['group_id']);selected.append(copy.deepcopy(c));taken+=1
                if taken==3:break
            assert taken==3,(family,domain,taken)
    controls('keep',dict(portrait=3,landscape=2))
    assert len(selected)==len({c['group_id'] for c in selected})==40
    assert Counter(c['domain'] for c in selected)==dict(portrait=20,landscape=20)
    for c in selected:
        assert sha(ROOT/c['image'])==c['input_sha256']
        if c['target']:assert sha(ROOT/c['target'])==c['target_sha256']
    native=sum(c['instruction_origin'].startswith('English untouched original GIER') for c in selected)
    write(destination,dict(at=stamp(),status='unchanged_prior_review_candidates_pending_final_replay_and_admission',
        cases=selected,semantic_cases=40,groups=40,domains=dict(portrait=20,landscape=20),
        original_English_cases=native,AI_control_cases=10,AI_edit_anchors=30-native,
        inherited_admission=str((PREVIOUS/'admission.json').relative_to(ROOT)),inherited_admission_sha256=sha(PREVIOUS/'admission.json'),
        inherited_cases_sha256=sha(PREVIOUS/'cases.json'),
        review_scope='Exactly unchanged Round7 input/request/target/criteria. Reuse prior direct visual observations explicitly; not40 newly independently reviewed images. Target replay and original-group isolation still required.',
        selection='Five single families x3 portrait/3 landscape tasks, prefer unchanged original English and explicitly use reviewed AI labels where that family/domain lacks originals;5 keep and5 unsupported controls balanced jointly.40 distinct photograph groups outside the20 new condition groups. No score/loss selection.'))
    print(f'Prepared40 unchanged anchors:{native} original-English edits,{30-native} AI edits,5 keep,5 unsupported;20/20domains.')


if __name__=='__main__':main()

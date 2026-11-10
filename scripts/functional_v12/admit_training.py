"""Exact reviewed R11 targets/requests; new manual never changes user instructions."""
import copy
import hashlib
from collections import Counter
import os
import numpy as np
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb
from .outcomes import input_row,execute,system_prompt,EDITOR_VERSION,SCHEMA_VERSION

EXP=ROOT/'experiments/functional-v12-20261106'
DATA=ROOT/'data/training/functional-v12-20261106'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (DATA/'admission.json').exists()
    old=ROOT/'data/training/functional-v11-20261106'
    assert sha(old/'cases.json')=='5604ab7aab492d3be0c0ca17f82c2329ec029a7b46174bee11a3d3e09fc6f359'
    lookup={c['id']:c for c in read(old/'cases.json')}
    source=ROOT/'data/training/functional-v10-20261106/cases.json'
    chosen=[c['id'].removeprefix('anchor-') for c in read(source)]
    chosen += [f'cool-cast-{i:02d}' for i in range(1,11)]
    assert len(chosen)==len(set(chosen))==250
    cases=[];replay=[];audit=[];rows=[];lineage=[]
    for cid in chosen:
        c=copy.deepcopy(lookup[cid]);assert c['split']=='train'
        original=copy.deepcopy(c)
        assert sha(ROOT/c['image'])==c['input_sha256']
        result=execute(load_rgb(ROOT/c['image']),c['decision'])
        if c['action']=='decline':assert result['result'] is None and c['target'] is None
        else:
            assert sha(ROOT/c['target'])==c['target_sha256']
            assert np.array_equal(result['result'],load_rgb(ROOT/c['target']))
        evidence=dict(kind='exact_reviewed_R11_request_and_target_inherited',source_case_id=cid,
            cases=str((old/'cases.json').relative_to(ROOT)),cases_sha256=sha(old/'cases.json'),
            admission=str((old/'admission.json').relative_to(ROOT)),admission_sha256=sha(old/'admission.json'),
            previous_label_evidence=original['label_evidence'],
            scope='Original user request, input, decision and target pixels unchanged; explicit prior AI evidence, not a new independent visual review. No new paraphrase or reversed request.')
        c.update(label_evidence=evidence,status='exact_reviewed_labels_inherited_for_new_tool_manual',
                 editor=EDITOR_VERSION,schema=SCHEMA_VERSION)
        assert all(c[k]==original[k] for k in ('instructions','image','input_sha256','decision','target','target_sha256'))
        cases.append(c);lineage.append(dict(case_id=cid,**evidence))
        replay.append(dict(case_id=cid,action=c['action'],input_sha256=c['input_sha256'],target_sha256=c['target_sha256'],exact_replay=True))
        for language in ('zh','en'):
            row=input_row(c,language)
            poisoned=dict(c,decision={'action':'keep'},target='FORBIDDEN_REFERENCE',parameters={'sepia':99},
                expected_change='FORBIDDEN_CRITERIA',label_evidence='FORBIDDEN_REVIEW',condition='FORBIDDEN_CONDITION',family='FORBIDDEN_FAMILY')
            assert input_row(poisoned,language)==row
            assert row['conversations']==[dict(role='system',content=system_prompt()),dict(role='user',content='<image>\n'+c['instructions'][language])]
            audit.append(dict(case_id=cid,language=language,metadata_poisoning_unchanged=True,
                input_sha256=hashlib.sha256(__import__('json').dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest()))
            rows.append(dict(id=cid+'-'+language,case_id=cid,language=language,domain=c['domain'],decision=c['decision'],instruction_origin=c['instruction_origin'],repeat=0))
    groups={c['group_id']:c['domain'] for c in cases}
    assert len(groups)==60 and Counter(groups.values())==dict(portrait=30,landscape=30)
    assert Counter(c['domain'] for c in cases)==dict(portrait=125,landscape=125)
    assert Counter(c['action'] for c in cases)==dict(edit=199,keep=5,decline=46)
    native=sum(c['instruction_origin'].startswith('English untouched original GIER') for c in cases);assert native==27
    isolation=[]
    for name,item in read(ROOT/'experiments/functional-v11-20261106/evaluation-protocol.json')['case_files'].items():
        file=ROOT/item['path'];assert sha(file)==item['sha256']
        assert not set(groups)&{c['group_id'] for c in read(file)}
        isolation.append(dict(partition=name,manifest=item,training_groups_disjoint=True,scope='Metadata hashes/group IDs only; original-source near-duplicate audit inherited. No heldout image or prediction opened.'))
    write(DATA/'cases.json',sorted(cases,key=lambda c:c['id']));write(DATA/'train-accepted.json',rows);write(DATA/'label-lineage.json',lineage)
    write(EXP/'target-replay.json',dict(at=stamp(),status='passed',records=replay))
    write(EXP/'input-audit.json',dict(at=stamp(),status='passed',records=audit,only_original_image_original_language_request_and_frozen_tools=True))
    write(EXP/'source-isolation.json',dict(at=stamp(),status='passed',records=isolation,groups=60,photo_domains=dict(Counter(groups.values()))))
    write(DATA/'admission.json',dict(at=stamp(),status='exact_prior_AI_reviewed_requests_and_targets_inherited',semantic_cases=250,
        unique_language_rows=500,actual_rows_each_route=500,source_groups=60,photo_domains=dict(Counter(groups.values())),
        case_domain_counts=dict(Counter(c['domain'] for c in cases)),outcome_counts=dict(Counter(c['action'] for c in cases)),
        original_GIER_English_cases=27,original_English_with_AI_Chinese_translation_rows=54,AI_authored_cases=223,AI_authored_rows=446,
        family_counts=dict(Counter(c['family'] for c in cases)),new_original_groups=0,parent_cumulative_original_groups=160,
        source_selection='Previous matched200-condition cases plus40 anchors, all taken from corrected R11 admission, plus10 existing cool-cast cases.250 distinct cases; no repetitions.',
        training_data_sha256=sha(DATA/'train-accepted.json'),cases_sha256=sha(DATA/'cases.json'),lineage_sha256=sha(DATA/'label-lineage.json'),
        prior_cases_sha256=sha(old/'cases.json'),prior_admission_sha256=sha(old/'admission.json'),
        limitations=['Every target/request is an exact prior reviewed R11 case, not fresh independent assessment.',
            'No new original photo, user upload, user feedback or heldout label enters training.',
            'New system manual describes unchanged editor semantics generally; it contains no heldout image, case ID or reference parameter.',
            'Synthetic conditions do not prove unknown real-camera reliability.'],GPU_hours=0,new_training_started=False))
    print('R12 exact reviewed250semantic500bilingual,60groups30/30;199edit5keep46decline;27original223AI. CPU only.')


if __name__=='__main__':main()

"""Admit only explicitly reviewed C/A labels; retain all failed attempts separately."""
import copy
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.editor import load_rgb
from functional_v11.outcomes import execute, input_row, validate_decision, EDITOR_VERSION, SCHEMA_VERSION
from .runtime import EXP, DATA


def notes(name):
    path=EXP/name
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    assert len(rows)==len({r['id'] for r in rows})
    return {r['id']:r for r in rows}


def evidence(bundle,raw_file,qa_file):
    manifest=EXP/bundle/'manifest.json'; bank={c['id']:c for c in read(manifest)['items']}
    raw=notes(raw_file); qa=notes(qa_file)
    assert set(raw)==set(bank)
    result={}
    for cid,n in raw.items():
        c=bank[cid]
        assert n['sheet']==c['sheet'] and n['sheet_sha256']==sha(ROOT/c['sheet'])==c['sheet_sha256']
        assert n['source_input_sha256']==c['input_sha256']==sha(ROOT/c['input_image'])
        selected=next((r for r in c['results'] if r['candidate']==n['chosen_candidate']),None)
        final=qa.get(cid,n)
        if cid in qa:
            q=qa[cid]
            assert q['chosen_candidate']==n['chosen_candidate'] and q['input_sha256']==c['input_sha256']
            assert sha(ROOT/q['manifest'])==q['manifest_sha256']
            assert q['output_sha256']==selected['image_sha256']
            for image in q['images']:assert sha(ROOT/image['path'])==image['sha256']
        assert n['quality']!='U' or cid in qa,('unresolved quality',cid)
        if selected:
            assert selected['image_sha256']==n['chosen_image_sha256']
            if selected['image']:assert sha(ROOT/selected['image'])==selected['image_sha256']
        result[cid]=(c,selected,dict(kind='direct_Codex_AI_review_not_human',
            raw_review=str((EXP/raw_file).relative_to(ROOT)),raw=n,
            native_pixel_supplement=qa.get(cid),completion=final['completion'],quality=final['quality'],
            manifest=str(manifest.relative_to(ROOT)),manifest_sha256=sha(manifest)))
    return result


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (DATA/'admission.json').exists()
    prior=ROOT/'data/training/functional-v11-20261106'
    assert sha(prior/'cases.json')=='5604ab7aab492d3be0c0ca17f82c2329ec029a7b46174bee11a3d3e09fc6f359'
    assert sha(prior/'train-accepted.json')=='fd998cd2bcf96a01f0312f17ba217e39ee593d04bdb1b73060cadc577f71328c'
    primary=evidence('primary-candidates','primary-observations.jsonl','primary-qa-observations.jsonl')
    repairs=evidence('primary-repairs','repair-observations.jsonl','repair-qa-observations.jsonl')
    pairs=evidence('paired-candidates','paired-observations.jsonl','paired-qa-observations.jsonl')
    selected={}
    for cid,value in primary.items():
        if value[2]['completion']=='C' and value[2]['quality']=='A':selected[cid]=value
    for cid,value in repairs.items():
        if value[2]['completion']=='C' and value[2]['quality']=='A':
            assert cid not in selected
            selected[cid]=value
    assert len(selected)==78
    for cid,value in pairs.items():
        if value[2]['completion']=='C' and value[2]['quality']=='A':selected[cid]=value
    cases=copy.deepcopy(read(prior/'cases.json')); parent_groups={c['group_id'] for c in cases}
    assert len(cases)==672 and len(parent_groups)==160
    for c in cases:
        # Preserve complete earlier evidence and unresolved historical limitations.
        c['r13_evidence']=dict(kind='exact_R11_record_inherited_not_new_visual_review',
            cases=str((prior/'cases.json').relative_to(ROOT)),cases_sha256=sha(prior/'cases.json'),
            admission=str((prior/'admission.json').relative_to(ROOT)),admission_sha256=sha(prior/'admission.json'))
    new=[]
    for cid,(item,result,proof) in selected.items():
        assert result is not None and item['group_id'] not in parent_groups
        assert sha(Path(item['source_image']))==item['source_sha256']
        origin=item['instruction_origin']
        if origin=='original_GIER_English_AI_Chinese':
            assert item['instructions']['en']==item['original_instruction']
        decision=validate_decision(result['decision'])
        c=dict(id='r13-'+cid,source_id=item.get('source_id',cid),group_id=item['group_id'],domain=item['domain'],split='train',
            family=item.get('family','global_combination'),image=item['input_image'],input_sha256=item['input_sha256'],
            instructions=item['instructions'],instruction_origin=origin,original_instruction=item['original_instruction'],
            native_annotation=item.get('native_annotation'),original_source_sha256=item['source_sha256'],
            canonical_photo=item['source_image'],source_partition='original GIER training pool; new to R9-R11 parent route, not new collection',
            condition_origin='original photograph; no synthetic pixel augmentation in this round',
            action=decision['action'],decision=decision,parameters=decision.get('parameters',{} if decision['action']=='keep' else None),
            target=result['image'],target_sha256=result['image_sha256'],editor=EDITOR_VERSION,schema=SCHEMA_VERSION,
            paired_direction=item.get('paired_direction'),status='direct_AI_admitted_not_human_or_user_approval',label_evidence=proof)
        new.append(c)
    cases+=new; assert len(cases)==len({c['id'] for c in cases})
    new_domains={c['group_id']:c['domain'] for c in new}
    assert len(new_domains)==80 and Counter(new_domains.values())==dict(portrait=40,landscape=40)
    new_ids={c['source_id'] for c in new}
    assert 'train-native-168' not in new_ids
    grouped=defaultdict(set)
    for c in new:
        if c.get('paired_direction'):grouped[c['group_id']].add(c['paired_direction'])
    complete_pairs=sum(v=={-1,1} for v in grouped.values()); assert complete_pairs>=60
    replay=[]; audit=[]; rows=[]
    for c in sorted(cases,key=lambda x:x['id']):
        assert sha(ROOT/c['image'])==c['input_sha256']
        decision=validate_decision(c['decision']); result=execute(load_rgb(ROOT/c['image']),decision)
        if decision['action']=='decline':assert result['result'] is None and c['target'] is None
        else:
            assert sha(ROOT/c['target'])==c['target_sha256']
            assert np.array_equal(result['result'],load_rgb(ROOT/c['target'])),c['id']
            if decision['action']=='keep':assert np.array_equal(result['result'],load_rgb(ROOT/c['image']))
        replay.append(dict(case_id=c['id'],input_sha256=c['input_sha256'],target_sha256=c['target_sha256'],exact_replay=True))
        for lang in ('zh','en'):
            row=input_row(c,lang)
            poison=dict(c,target='FORBIDDEN_TARGET',decision={'action':'keep'},parameters={'sepia':99},family='FORBIDDEN_TASK',
                label_evidence='FORBIDDEN_REVIEW',r13_evidence='FORBIDDEN_HISTORY',expected_change='FORBIDDEN_REQUIREMENT',
                condition_origin='FORBIDDEN_CONDITION',native_annotation='FORBIDDEN_ANNOTATION',paired_direction='FORBIDDEN_DIRECTION')
            assert input_row(poison,lang)==row
            audit.append(dict(case_id=c['id'],language=lang,metadata_poisoning_unchanged=True,
                input_sha256=hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest()))
            rows.append(dict(id=c['id']+'-'+lang,case_id=c['id'],language=lang,domain=c['domain'],decision=decision,
                instruction_origin=c['instruction_origin'],repeat=0))
    for name,item in read(ROOT/'experiments/functional-v12-20261106/evaluation-protocol.json')['case_files'].items():
        assert sha(ROOT/item['path'])==item['sha256']
        assert not (parent_groups|set(new_domains)) & {c['group_id'] for c in read(ROOT/item['path'])},name
    iso=read(EXP/'source-isolation-extended.json'); assert iso['flags']==[]
    excluded=[]
    for bundle,bank in [('primary',primary),('repair',repairs),('paired',pairs)]:
        for cid,(_,_,p) in bank.items():
            if p['completion']!='C' or p['quality']!='A':excluded.append(dict(bundle=bundle,id=cid,proof=p))
    write(DATA/'cases.json',sorted(cases,key=lambda c:c['id']));write(DATA/'train-accepted.json',rows)
    write(DATA/'excluded-attempts.json',excluded)
    write(EXP/'target-replay.json',dict(at=stamp(),status='passed',records=replay))
    write(EXP/'input-audit.json',dict(at=stamp(),status='passed',records=audit,only_image_literal_request_tools=True))
    source_files=[p for p in EXP.glob('*observations.jsonl')]+[EXP/'source-amendments.jsonl',EXP/'source-isolation-extended.json']
    source_files += [EXP/b/'manifest.json' for b in ('primary-candidates','primary-repairs','paired-candidates')]
    native=lambda c:c['instruction_origin']=='original_GIER_English_AI_Chinese' or c['instruction_origin'].startswith('English untouched original GIER')
    write(DATA/'admission.json',dict(at=stamp(),status='direct_AI_visual_admitted_not_human_or_user_approval',
        semantic_cases=len(cases),unique_language_rows=len(rows),actual_rows_each_route=len(rows),source_groups=240,
        photo_domains=dict(portrait=120,landscape=120),case_domains=dict(Counter(c['domain'] for c in cases)),
        new_semantic_cases=len(new),inherited_R11_cases=672,new_original_groups=80,new_photo_domains=dict(Counter(new_domains.values())),
        complete_new_opposite_request_pairs=complete_pairs,new_outcomes=dict(Counter(c['action'] for c in new)),
        outcomes=dict(Counter(c['action'] for c in cases)),original_English_cases=sum(map(native,cases)),
        new_original_English_cases=sum(map(native,new)),AI_authored_cases=sum(not native(c) for c in cases),
        excluded_failed_attempts=len(excluded),new_pair_failures=[x['id'] for x in excluded if x['bundle']=='paired'],
        training_data_sha256=sha(DATA/'train-accepted.json'),cases_sha256=sha(DATA/'cases.json'),
        evidence_hashes={str(p.relative_to(ROOT)):sha(p) for p in source_files},
        frozen_Qwen_screening='Not used as admission judge: R8 frozen-critic calibration failed. This is an explicit deviation from initial screening plan, not independent human validation.',
        limitations=['All new photographs are existing original training-pool photographs, new only to the R9-R11 parent route.',
            'Existing672 records inherit exact R11 evidence and its unresolved limitations; not re-reviewed as new independent labels.',
            'All new labels have direct Codex AI image review; no human approval or actual user feedback.',
            'Failed edits are excluded, not automatically turned into universal impossibility/refusal labels.',
            'Original English and AI Chinese translations are one semantic case; AI requests and reversals are explicitly identified.',
            'Near-duplicate checks cover source metadata, hashes and image fingerprints; related captures can still escape detection.'],
        no_trial_upload_or_feedback_in_training=True,no_GPU_backprop_started=True))
    print(json.dumps(read(DATA/'admission.json'),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

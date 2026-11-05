"""Admit only directly reviewed executed targets; freeze the actual matched experiment."""
import copy
import hashlib
import json
from collections import Counter
import numpy as np
from functional_v1.common import ROOT,read,write,sha,stamp,input_row,system_prompt,EDITOR_VERSION
from functional_v1.editor import edit,load_rgb
from functional_v3.common import PARENT
from .source_bank import EXP,DATA


def main():
    assert not (EXP/'training-protocol.json').exists()
    original=read(DATA/'native-training-candidates.json')
    initial=list(map(json.loads,(EXP/'native-label-review/observations.jsonl').read_text().splitlines()))
    repairs=read(EXP/'native-label-revision/observations.json')
    amendment=read(EXP/'native-cohort-amendment/amendment.json')
    additions=read(DATA/'native-training-additions.json')
    added_notes=read(EXP/'native-cohort-amendment/observations.json')
    assert len(initial)==100 and len(repairs)==22 and len(added_notes)==3
    assert sha(DATA/'native-training-additions.json')==amendment['additions_sha256']
    selected={r['case_id']:r for r in initial if r['selected'] is not None}
    for r in repairs:
        assert r['case_id'] not in selected
        if r['selected'] is not None:selected[r['case_id']]=r
    detail=read(EXP/'native-label-detail/observations.json');detail_proposal=read(EXP/'native-label-detail/proposal.json')
    assert sha(EXP/'native-label-detail/proposal.json')==detail['proposal_sha256']
    assert detail['completion']=='completed' and detail['quality']=='acceptable' and detail['language_equivalence_checked']
    assert detail['case_id']==detail_proposal['case_id']=='native-train-n048'
    assert sha(ROOT/detail_proposal['image'])==detail['image_sha256']==detail_proposal['sha256']
    selected[detail['case_id']]=dict(case_id=detail['case_id'],selected_code='G',language_equivalence_checked=True,
        input_sha256=detail['input_sha256'],candidates={'G':detail},
        selected=dict(parameters=detail_proposal['parameters'],image=detail_proposal['image'],sha256=detail_proposal['sha256'],proposal_route='root_authored_full_resolution_repair'))
    assert len(selected)==97 and set(amendment['excluded'])=={c['id'] for c in original}-{*selected}
    choices={r['case_id']:r for r in amendment['target_proposals']}
    for note in added_notes:
        cid=note['case_id'];code=note['selected_code']
        assert note['language_equivalence_checked'] and set(note['candidates'])=={'E','F'}
        assert sha(ROOT/amendment['panel'])==amendment['panel_sha256']
        for assessment in note['candidates'].values():
            assert assessment['completion'] in ('completed','partial','failed','uncertain')
            assert assessment['quality'] in ('acceptable','minor_issue','major_issue','uncertain')
            assert len(assessment['evidence'])>=15
        assessment=note['candidates'][code]
        assert assessment['completion']=='completed' and assessment['quality']=='acceptable'
        selected[cid]=dict(note,selected=dict(choices[cid]['candidates'][code],proposal_route='root_authored_source_amendment'),
            input_sha256=choices[cid]['input_sha256'])
    native=[]
    for raw in original+additions:
        if raw['id'] in amendment['excluded']:continue
        c=copy.deepcopy(raw);note=selected[c['id']];target=note['selected']
        assert note['language_equivalence_checked'] and note['input_sha256']==c['input_sha256']
        assert sha(ROOT/c['image'])==c['input_sha256'] and sha(ROOT/target['image'])==target['sha256']
        c.update(parameters=target['parameters'],target=target['image'],target_sha256=target['sha256'],
            target_status='direct_AI_visual_admitted_not_user_approval',target_origin=target['proposal_route'],
            direct_label_evidence=note['candidates'][note['selected_code']]['evidence'])
        native.append(c)
    assert len(native)==len({c['group_id'] for c in native})==100
    assert Counter(c['domain'] for c in native)=={'portrait':50,'landscape':50}
    prior_data=ROOT/'data/training/functional-v4-20261105'
    controlled=[c for c in read(prior_data/'cases.json') if c['action']!='original-natural-request']
    assert len(controlled)==360 and len({c['group_id'] for c in controlled})==60
    cases=controlled+native
    assert len(cases)==460 and len({c['id'] for c in cases})==460
    assert len({c['group_id'] for c in cases})==160
    for file in (DATA/'development-native-cases.json',DATA/'acceptance-native-cases.json',
            ROOT/'data/training/functional-v2-20261104/development-final-cases.json',
            ROOT/'data/training/functional-v2-20261104/acceptance-final-cases.json'):
        assert not {c['group_id'] for c in cases}&{c['group_id'] for c in read(file)}
    replay=[];audit=[];rows=[]
    for c in cases:
        assert c['split']=='train' and sha(ROOT/c['image'])==c['input_sha256']
        assert sha(ROOT/c['target'])==c['target_sha256']
        assert np.array_equal(edit(load_rgb(ROOT/c['image']),c['parameters']),load_rgb(ROOT/c['target']))
        replay.append(dict(case_id=c['id'],input_sha256=c['input_sha256'],target_sha256=c['target_sha256'],pixel_identical=True))
        for lang in ('zh','en'):
            r=input_row(c,lang)
            poisoned=dict(c,parameters={'exposure':-99},target='FORBIDDEN_TARGET_SENTINEL',reference='FORBIDDEN_REFERENCE_SENTINEL',expected_change='FORBIDDEN_REVIEW_SENTINEL',family='FORBIDDEN_FAMILY_SENTINEL')
            assert input_row(poisoned,lang)==r
            assert r['conversations']==[{'role':'system','content':system_prompt()},{'role':'user','content':'<image>\n'+c['instructions'][lang]}]
            audit.append(dict(case_id=c['id'],language=lang,metadata_poisoning_unchanged=True,
                serialized_input_sha256=hashlib.sha256(json.dumps(r,sort_keys=True,ensure_ascii=False).encode()).hexdigest()))
            rows.append(dict(id=c['id']+'-'+lang,case_id=c['id'],language=lang,parameters=c['parameters'],domain=c['domain'],
                origin='original native English / AI Chinese translation' if c in native else 'AI-authored controlled request',repeat=0))
    assert len(rows)==len({r['id'] for r in rows})==len({(r['case_id'],r['language']) for r in rows})==920
    write(DATA/'cases.json',cases);write(DATA/'train-accepted.json',rows);write(EXP/'target-replay.json',replay)
    write(EXP/'input-audit.json',dict(status='passed',at=stamp(),rows=audit,
        only_input_image_original_single_language_request_and_tools=True,hidden_metadata_poisoning_cannot_change_input=True))
    write(DATA/'admission.json',dict(at=stamp(),status='direct_AI_visual_admitted_not_user_approval',
        semantic_cases=460,native_original_cases=100,controlled_AI_authored_cases=360,source_groups=160,
        photo_domains={'portrait':80,'landscape':80},actual_rows_each_route=920,unique_language_rows=920,
        native_language_rows=200,controlled_language_rows=720,oversampling=False,
        native_first_cohort=100,native_first_proposal_selected=78,authored_repairs_selected=19,
        initial_sources_excluded_after_failed_labels=amendment['excluded'],replacement_native_sources=3,
        native_sources_considered_for_target_admission=103,
        target_origins=dict(Counter(c['target_origin'] for c in native)),
        cases_sha256=sha(DATA/'cases.json'),dataset_sha256=sha(DATA/'train-accepted.json'),
        native_evidence_hashes={str(f.relative_to(ROOT)):sha(f) for f in [EXP/'native-label-review/observations.jsonl',EXP/'native-label-revision/observations.json',EXP/'native-cohort-amendment/amendment.json',EXP/'native-cohort-amendment/observations.json',EXP/'native-label-detail/observations.json',EXP/'native-label-detail/proposal.json']},
        controlled_lineage='360 round2 directly reviewed targets with six round3 corrections, reused unchanged from round4. Ten repeatedly oversampled round4 native tasks removed.',
        controlled_admission_sha256=sha(prior_data/'admission.json'),
        limitations='AI-only target judgement, not human approval. Training-source replacement is purposeful curation and cannot estimate model success. Historical source pools and earlier parents may have seen training photos. Heldout requests/gates unchanged.'))
    core=read(EXP/'native-proposal-protocol.json')
    for file,digest in core['code_sha256'].items():assert sha(ROOT/file)==digest
    evaluation=read(EXP/'evaluation-protocol.json')
    for entry in evaluation['case_files'].values():assert sha(ROOT/entry['path'])==entry['sha256']
    settings=dict(epochs=4,learning_rate=1e-4,effective_batch=4,steps=920,steps_per_epoch=230,seed=20261110,
        replicate_seed=20261111,rank=8,alpha=16,dropout=0.,quantization='NF4, double quant, BF16 compute',
        modules='language q/v only;112 trainable FP32 LoRA tensors, vision frozen',per_device_batch=1,
        gradient_accumulation=4,warmup_ratio=.05,scheduler='cosine',optimizer='adamw_torch',weight_decay=0.,
        max_grad_norm=1.,save='every complete epoch',rows=920)
    files=['scripts/functional_v1/common.py','scripts/functional_v1/runtime.py','scripts/functional_v1/editor.py',
        'scripts/functional_v6/train.py','scripts/functional_v6/prepare_training.py']
    write(EXP/'training-protocol.json',dict(at=stamp(),status='actual_data_targets_runtime_optimizer_frozen_before_training',
        hypothesis='Add100 unique source-audited native-request examples across100 extra photos; remove tiny-set oversampling and measured-input note. Matched fresh-base vs continued-parent QLoRA isolates initialization under the same920 distinct bilingual records.',
        routes=['base','continued'],parent=str(PARENT.relative_to(ROOT)),parent_sha256=sha(PARENT/'adapter_model.safetensors'),
        dataset_sha256=sha(DATA/'train-accepted.json'),cases_sha256=sha(DATA/'cases.json'),admission_sha256=sha(DATA/'admission.json'),
        input_audit_sha256=sha(EXP/'input-audit.json'),target_replay_sha256=sha(EXP/'target-replay.json'),
        code_hashes={f:sha(ROOT/f) for f in files},training=settings,editor=EDITOR_VERSION,
        evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),
        frozen_runtime='functional_v1.runtime, exact v1 prompt; no round4 quality NOTE/statistics, no round5 head',
        system_prompt_sha256=hashlib.sha256(system_prompt().encode()).hexdigest(),
        selection='Functional direct-image gates in frozen evaluation-protocol.json only; no L1/parameter error selection.',
        gpu_budget='Original shared16 occupied GPU hours, GPU0/2/3 only; protect GPU1.'))
    print(json.dumps(dict(cases=460,groups=160,rows=920,dataset_sha256=sha(DATA/'train-accepted.json'))))

if __name__=='__main__':main()

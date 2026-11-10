"""Admit only directly inspected targets and explicit limited-tool decisions.

Old failed pixels stay unchanged. The selected repairs are different executed
images, each reviewed at original size. No loss or numeric image score ranks
labels, and synthetic conditions are never counted as independent photos.
"""
import copy
import hashlib
import json
from collections import Counter,defaultdict
import numpy as np
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb
from functional_v7.outcomes import execute,input_row,system_prompt,validate_decision,EDITOR_VERSION,SCHEMA_VERSION
from .prepare_counterfactuals import EXP,DATA,resize,noise,render


def lines(path):return [json.loads(line) for line in path.read_text().splitlines()]
def relative(path):return str(path.relative_to(ROOT))


def completion_quality(value):
    """Read historical note formats without discarding an uncertainty override."""
    if isinstance(value, dict):
        return [value.get('completion', value.get('full_completion')),
                value.get('effective_quality', value.get('quality', value.get('full_quality')))]
    assert isinstance(value, list) and len(value)==2, value
    return value


def main():
    assert not (DATA/'admission.json').exists(),'Preserve admitted data'
    prior=ROOT/'data/training/functional-v7-20261105'
    assert sha(prior/'cases.json')=='1948fbe5e40cdeae1d3e37fefedd4c87d9ebb7a2cf1eec3ff830229e8b98353a'
    originals=read(prior/'cases.json');allowed={c['group_id']:c['domain'] for c in originals}
    assert len(allowed)==160 and Counter(allowed.values())==dict(portrait=80,landscape=80)
    source=read(EXP/'counterfactual-candidates/manifest.json')
    ext={c['id']:c for c in read(EXP/'extended-candidates/manifest.json')['cases']}
    style={c['id']:c for c in read(EXP/'style-candidates/manifest.json')['cases']}
    dark={c['id']:c for c in read(EXP/'darkening-candidates/manifest.json')['cases']}
    initial={c['id']:c for c in lines(EXP/'counterfactual-candidates/direct-observations.jsonl')}
    stages={(c['id'],c['stage']):c for c in lines(EXP/'candidate-stage-observations.jsonl')}
    full={c['id']:c for c in lines(EXP/'candidate-full-resolution-observations.jsonl')}
    assert len(source['cases'])==len(initial)==len(full)==20 and len(stages)==60
    provenance_files=[prior/'admission.json',prior/'cases.json',EXP/'anchor-candidates.json',
        EXP/'counterfactual-candidates/manifest.json',EXP/'extended-candidates/manifest.json',
        EXP/'style-candidates/manifest.json',EXP/'darkening-candidates/manifest.json',
        EXP/'counterfactual-candidates/direct-observations.jsonl',EXP/'candidate-stage-observations.jsonl',
        EXP/'candidate-full-resolution-observations.jsonl']
    repairs={}
    for folder in ('reviewed-portrait-batch-03-07-08-09','reviewed-landscape-batch-12-15-17-18-20'):
        path=EXP/'repairs'/folder;manifest=read(path/'manifest.json');review=read(path/'direct-review.json')
        assert sha(path/'manifest.json')==review['manifest_sha256']
        notes={n['id']:n for n in review['judgments']}
        for c in manifest['candidates']:
            n=notes[c['id']]
            if n['selected_replacement']:
                assert [n['completion'],n['quality']]==['completed','acceptable'] and n['output']==c['output']
                repairs[(c['group'],c['role'])]=dict(parameters=c['parameters'],output=c['output'],
                    evidence=n['evidence'],review=relative(path/'direct-review.json'),input=c['input'])
        provenance_files.extend([path/'manifest.json',path/'direct-review.json'])
    p=EXP/'repairs/portrait-normal-compound-02';m=read(p/'manifest.json');n=read(p/'direct-review.json')
    assert sha(p/'manifest.json')==n['manifest_sha256']
    c=n['selected_candidate'];note=next(r for r in n['observations'] if r['id']==c['id'])
    assert [note['completion'],note['quality']]==['completed','acceptable'] and c in m['candidates']
    repairs[('visual-risk-02','normal-compound')]=dict(parameters=c['parameters'],
        output=dict(path=c['image'],sha256=c['sha256']),evidence=note['evidence'],review=relative(p/'direct-review.json'))
    provenance_files.extend([p/'manifest.json',p/'direct-review.json'])
    p=EXP/'repairs/portrait-white-detail-07';white=read(p/'direct-review.json')
    assert sha(p/'manifest.json')==white['manifest_sha256'] and white['selected_replacement'] is None
    assert white['proposed_decision']==dict(action='decline',reason='unreliable_quality')
    assert not any(n['completion']=='completed' and n['quality']=='acceptable' for n in white['judgments'])
    provenance_files.extend([p/'manifest.json',p/'direct-review.json'])
    cases=[];label_notes=[];source_replay=[]

    def add(c,decision,target,evidence):
        decision=validate_decision(decision)
        c.update(action=decision['action'],decision=decision,parameters=decision.get('parameters',{} if decision['action']=='keep' else None),
            target=target['path'] if target else None,target_sha256=target['sha256'] if target else None,
            editor=EDITOR_VERSION,schema=SCHEMA_VERSION,status='direct_AI_admitted_not_human_or_user_approval',label_evidence=evidence)
        assert c['group_id'] in allowed and c['domain']==allowed[c['group_id']] and c['split']=='train'
        cases.append(c);label_notes.append(dict(case_id=c['id'],action=c['action'],**evidence))

    for index,c in enumerate(source['cases']):
        cid=c['id'];e=ext[cid];s=style[cid];d=dark[cid]
        assert c['group_id'] in allowed and all(x['group_id']==c['group_id'] for x in (e,s,d))
        assert sha(ROOT/c['canonical_photo'])==c['canonical_sha256']
        known=[o for o in originals if o['group_id']==c['group_id'] and o.get('canonical_photo')==c['canonical_photo']]
        assert known and all(o['original_source_sha256']==c['original_source_sha256'] for o in known)
        canonical=resize(load_rgb(ROOT/c['canonical_photo']))
        assert np.array_equal(canonical,load_rgb(ROOT/c['images']['canonical']['path']))
        deep=render(canonical,dict(exposure=-65))
        severe,corruption=noise(deep,index)
        mild=np.clip(np.rint(deep.astype(np.float32)+np.random.default_rng(20261120+index).normal(0,.55,deep.shape)),0,255).astype(np.uint8)
        expected={'deep-clean-input':deep,'noisy-input':severe,'mild-noise-input':mild,
                  'bright-shallow-input':render(canonical,dict(exposure=30))}
        images={**c['images'],'mild-noise-input':e['images']['mild-noise-input'],'bright-shallow-input':d['images']['bright-shallow-input']}
        for role,pixels in expected.items():assert np.array_equal(pixels,load_rgb(ROOT/images[role]['path'])),(cid,role)
        assert corruption==c['corruption']
        source_replay.append(dict(id=cid,original_group=c['group_id'],canonical_hash=c['canonical_sha256'],
            resized_before_augmentation=True,deep_mild_severe_bright_conditions_replayed_exactly=True,synthetic_noise=True))
        assert initial[cid]['clean_deep']['completion']=='completed' and initial[cid]['clean_deep']['quality']=='acceptable'
        assert initial[cid]['initial_noise_attempts']['quality']=='major_issue'
        for role in ('severe-light-damped','severe-light-shadow-damped'):
            assert stages[cid,'light-alternatives']['judgments'][role][1]=='major_issue'
        for role in ('severe-compound','severe-compound-gentle','severe-compound-shadows'):
            assert stages[cid,'compound']['judgments'][role][1]=='major_issue'
        plans=[('deep-light','deep-clean-input','light',c['images']['deep-clean-edit'],dict(exposure=65),'initial'),
            ('normal-light','canonical','light',s['images']['normal-light'],s['parameters']['normal-light'],'normal-and-dark-style'),
            ('mild-noise-light','mild-noise-input','light',e['images']['mild-noise-light'],e['settings']['mild-noise-light'],'light-alternatives'),
            ('severe-light','noisy-input','light',None,None,'quality-decline'),
            ('deep-compound','deep-clean-input','compound',e['images']['deep-compound'],e['settings']['deep-compound'],'compound'),
            ('normal-compound','canonical','compound',s['images']['normal-compound'],s['parameters']['normal-compound'],'normal-and-dark-style'),
            ('mild-noise-compound','mild-noise-input','compound',e['images']['mild-noise-compound'],e['settings']['mild-noise-compound'],'compound'),
            ('severe-compound','noisy-input','compound',None,None,'quality-decline'),
            ('normal-dark-style','canonical','dark',s['images']['normal-dark-style'],s['parameters']['normal-dark-style'],'normal-and-dark-style'),
            ('bright-dark-style','bright-shallow-input','dark',s['images']['bright-dark-style'],s['parameters']['bright-dark-style'],'normal-and-dark-style')]
        for role,input_role,request,target,parameters,stage in plans:
            inp=images[input_role];decision=dict(action='edit',parameters=parameters)
            evidence=dict(kind='new_direct_visual_review',source_group=cid,stage=stage,
                full_resolution_review=relative(EXP/'candidate-full-resolution-observations.jsonl'),
                panel_review=relative(EXP/('counterfactual-candidates/direct-observations.jsonl' if stage=='initial' else 'candidate-stage-observations.jsonl')))
            if stage=='quality-decline':
                decision=dict(action='decline',reason='unreliable_quality')
                evidence.update(kind='direct_AI_quality_decline_after_actual_failed_alternatives',
                    evidence='Initial lifts and compensated/gentle attempts retained intrusive coloured blocks/bands. With the current global tools, a reliable result meeting this request was not found. Original corruption and amplified visibility are distinguished.',
                    limitation='Synthetic corruption, not real sensor capture. Tried alternatives do not prove every possible parameter combination impossible. No output target for decline.')
            elif cid=='visual-risk-07' and role=='normal-light':
                decision=dict(action='decline',reason='unreliable_quality');target=None
                evidence.update(kind='direct_AI_quality_decline_after_white_detail_repairs',review=relative(p/'direct-review.json'),
                    evidence='Already-normal bright-white portrait: strong lifts flatten lace/white detail; the more moderate candidate does not reliably complete substantial brightening. Other requests on the same input remain editable.',
                    limitation='AI judgment of current global tools and inspected attempts, not a universal impossibility claim.')
            elif (cid,role) in repairs:
                r=repairs[cid,role];assert r['output']['sha256']!=target['sha256']
                if 'input' in r:assert r['input']['path']==inp['path'] and r['input']['sha256']==inp['sha256']
                target=r['output'];decision=dict(action='edit',parameters=r['parameters'])
                evidence.update(kind='direct_original_size_reviewed_replacement',review=r['review'],evidence=r['evidence'],old_pixels_excluded=True)
            else:
                if stage!='initial':assert stages[cid,stage]['judgments'][role]==['completed','acceptable'],(cid,role)
                assert completion_quality(full[cid].get('judgments',{}).get(role,['completed','acceptable']))==['completed','acceptable'],(cid,role,'full-size concern')
                evidence['evidence']=initial[cid]['clean_deep']['evidence'] if stage=='initial' else stages[cid,stage]['evidence']
            raw=dict(id=cid+'-'+role,source_id=c['source_id'],group_id=c['group_id'],domain=c['domain'],split='train',
                family='compound' if request=='compound' else 'lighting',image=inp['path'],input_sha256=inp['sha256'],
                instructions=s['requests'][request],expected_change=s['requests'][request]['zh'],
                must_preserve='主体、构图、原有颜色（除明确要求的鲜明度变化）、天空与暗部细节；避免明显噪声放大、过曝、压黑、额外偏色。',
                failure_conditions=['要求的方向或力度未完成','复合要求漏项','明显画质副作用','可行任务被无故拒绝'],
                instruction_origin='AI-authored matched photographic-condition request in Chinese and English; not an untouched GIER request or user trial.',
                source_partition='original training_pool; existing Round7 training group',original_source_sha256=c['original_source_sha256'],
                canonical_photo=c['canonical_photo'],condition=input_role,
                condition_origin='canonical resized copy' if input_role=='canonical' else 'explicit AI-programmed synthetic exposure/noise condition',
                synthetic_corruption=c['corruption'] if input_role=='noisy-input' else dict(kind='Gaussian .55 uint8 LSB',seed=20261120+index,synthetic=True) if input_role=='mild-noise-input' else None)
            add(raw,decision,target,evidence)
    anchors=read(EXP/'anchor-candidates.json')
    for raw in anchors['cases']:
        previous=next(o for o in originals if o['id']==raw['id']);assert raw==previous
        target=dict(path=raw['target'],sha256=raw['target_sha256']) if raw['target'] else None
        c=copy.deepcopy(raw);c['id']='anchor-'+raw['id']
        add(c,raw['decision'],target,dict(kind='exact_unchanged_prior_reviewed_anchor',source_case_id=raw['id'],
            previous_admission=anchors['inherited_admission'],previous_admission_sha256=anchors['inherited_admission_sha256'],
            previous_label_evidence=raw['label_evidence'],review_scope='Exact prior pixels/request/criteria reused explicitly; not a new independent review.'))
    assert len(cases)==len({c['id'] for c in cases})==240
    groups={c['group_id']:c['domain'] for c in cases}
    assert len(groups)==60 and Counter(groups.values())==dict(portrait=30,landscape=30)
    assert Counter(c['domain'] for c in cases)==dict(portrait=120,landscape=120)
    isolation=[]
    for name,item in read(ROOT/'experiments/functional-v9-20261106/evaluation-protocol.json')['case_files'].items():
        file=ROOT/item['path'];assert sha(file)==item['sha256']
        held=read(file);assert not set(groups)&{c['group_id'] for c in held},name
        isolation.append(dict(partition=name,manifest=item,training_groups_disjoint=True,
            scope='Read metadata and verify existing group IDs only; no heldout images or predictions opened. Near-duplicate exclusion inherits the earlier160-group original-source audit.'))
    replay=[];audit=[];rows=[];conditioning=defaultdict(set)
    for c in cases:
        assert sha(ROOT/c['image'])==c['input_sha256'];rgb=load_rgb(ROOT/c['image']);result=execute(rgb,c['decision'])
        if c['action']=='decline':assert result['result'] is None and result['status']=='declined' and c['target'] is None
        else:
            assert sha(ROOT/c['target'])==c['target_sha256']
            assert np.array_equal(result['result'],load_rgb(ROOT/c['target'])),c['id']
            if c['action']=='keep':assert np.array_equal(rgb,result['result']) and result['status']=='unchanged'
        replay.append(dict(case_id=c['id'],action=c['action'],status=result['status'],input_sha256=c['input_sha256'],target_sha256=c['target_sha256'],exact_replay=True))
        conditioning[c['instructions']['en']].add(c['action'])
        for language in ('zh','en'):
            row=input_row(c,language)
            poisoned=dict(c,decision={'action':'keep'},parameters={'sepia':99},target='FORBIDDEN_TARGET',reference='FORBIDDEN_REFERENCE',
                expected_change='FORBIDDEN_REVIEW',condition='FORBIDDEN_CONDITION',synthetic_corruption='FORBIDDEN_NOISE',label_evidence='FORBIDDEN_LABEL',family='FORBIDDEN_FAMILY')
            assert input_row(poisoned,language)==row
            assert row['conversations']==[dict(role='system',content=system_prompt()),dict(role='user',content='<image>\n'+c['instructions'][language])]
            audit.append(dict(case_id=c['id'],language=language,metadata_poisoning_unchanged=True,input_hash=hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest()))
            rows.append(dict(id=c['id']+'-'+language,case_id=c['id'],language=language,domain=c['domain'],decision=c['decision'],instruction_origin=c['instruction_origin'],repeat=0))
    assert len(rows)==len({r['id'] for r in rows})==480
    assert sum(v=={'edit','decline'} for v in conditioning.values())==2
    assert Counter(c['action'] for c in cases)==dict(edit=189,keep=5,decline=46)
    native=sum(c['instruction_origin'].startswith('English untouched original GIER') for c in cases);assert native==27
    write(DATA/'cases.json',sorted(cases,key=lambda c:c['id']));write(DATA/'train-accepted.json',rows)
    write(DATA/'label-lineage.json',label_notes)
    write(EXP/'source-replay.json',dict(at=stamp(),status='passed',records=source_replay))
    write(EXP/'source-isolation.json',dict(at=stamp(),status='passed',records=isolation,groups=60,cumulative_parent_training_groups=160,all_within_prior_training_pool=True))
    write(EXP/'target-replay.json',dict(at=stamp(),status='passed',records=replay))
    write(EXP/'input-audit.json',dict(at=stamp(),status='passed',records=audit,only_input_image_one_request_and_tool_prompt=True))
    write(DATA/'admission.json',dict(at=stamp(),status='direct_AI_visual_admitted_not_human_or_user_approval',
        semantic_cases=240,unique_language_rows=480,actual_rows_each_route=480,source_groups=60,photo_domains=dict(Counter(groups.values())),
        cumulative_parent_source_groups=160,new_photograph_groups_added=0,new_AI_semantic_cases=200,unchanged_anchor_cases=40,
        outcome_counts=dict(Counter(c['action'] for c in cases)),new_quality_declines=41,unsupported_control_declines=5,
        original_GIER_English_cases=native,original_English_with_AI_Chinese_translation_rows=native*2,
        AI_authored_cases=240-native,AI_authored_rows=(240-native)*2,original_English_semantic_fraction=native/240,
        family_counts=dict(Counter(c['family'] for c in cases)),case_domain_counts=dict(Counter(c['domain'] for c in cases)),
        same_exact_requests_with_both_edit_and_quality_decline=2,synthetic_noise_is_not_real_camera_noise=True,
        training_data_sha256=sha(DATA/'train-accepted.json'),cases_sha256=sha(DATA/'cases.json'),lineage_sha256=sha(DATA/'label-lineage.json'),
        provenance_hashes={relative(p):sha(p) for p in provenance_files},
        selection='Direct viewed completion plus basic image quality, explicit repair choices and no L1/reward ranking. Bad/uncertain old pixels excluded and preserved.',
        frozen_Qwen_screening='Earlier directional calibration failed; not trusted to admit these labels. Codex direct visual AI review is the final source, not human approval.',
        limitations=['200 new semantic cases use only20 existing original groups and3 AI request templates with recorded synthetic conditions; not independent photos.',
            '40 anchors inherit exact prior direct review, including27 original-English requests and3 AI white-balance landscape edits; controls10AI.',
            'Quality declines reflect current global tools and tried alternatives, not exhaustive impossibility proofs or a guarantee of real-photo refusal calibration.',
            'No user-uploaded photograph or feedback enters training; independent user feedback remains zero.'],
        no_GPU_training_started=True))
    print('Admitted240semantic/480bilingual,60groups30/30;189edit5keep46decline;27original-English/213AI. All source/target/input audits passed.')


if __name__=='__main__':main()

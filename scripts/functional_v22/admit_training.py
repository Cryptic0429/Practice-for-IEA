"""Admit only the64 actually viewed and natively resolved TRAIN conditions."""
import collections,json,shutil
import numpy as np
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb
from functional_v15.outcomes import validate_decision,execute,EDITOR_VERSION,SCHEMA_VERSION
from .prepare_quality_data import EXP,DATA


def main():
    assert not (DATA/'admission.json').exists()
    plan=read(EXP/'quality-data-plan.json');source=ROOT/plan['source_cases']
    assert sha(source)==plan['source_sha256'];old=read(source);assert len(old)==925
    manifest_path=EXP/'quality-candidates/manifest.json';manifest=read(manifest_path)
    native_path=EXP/'quality-native-review/manifest.json';native=read(native_path)
    triage_path=EXP/'quality-triage-notes.jsonl';notes_path=EXP/'quality-native-notes.jsonl'
    triage={r['id']:r for r in map(json.loads,triage_path.read_text().splitlines())}
    notes={r['id']:r for r in map(json.loads,notes_path.read_text().splitlines())}
    witnesses={r['id']:r for r in native['rows']}
    assert len(triage)==len(notes)==len(witnesses)==len(manifest['items'])==64
    assert native['candidate_manifest_sha256']==sha(manifest_path) and native['triage_notes_sha256']==sha(triage_path)
    previous=read(ROOT/'experiments/functional-v21-20261107/training-protocol.json');groups={c['group_id'] for c in old}
    for spec in list(previous['sources'].values())+list(previous['sealed_acceptance_sources'].values()):
        assert sha(ROOT/spec['path'])==spec['sha256']
        if spec['path']!=plan['source_cases']:assert not groups&{c['group_id'] for c in read(ROOT/spec['path'])}
    additions=[];excluded=[]
    for c in manifest['items']:
        n=notes[c['id']];w=witnesses[c['id']];t=triage[c['id']]
        assert sha(ROOT/t['panel'])==t['panel_sha256']==c['panel']['sha256'] and t['viewed_all3_candidates']
        assert sha(ROOT/n['panel']['path'])==n['panel']['sha256']==w['panel']['sha256']
        assert n['candidate_id']==w['candidate_id'] and n['input_sha256']==c['input_sha256']
        assert sha(ROOT/c['image'])==c['input_sha256']
        if n['disposition']=='exclude_uncertain':
            excluded.append(dict(id=c['id'],group_id=c['group_id'],domain=c['domain'],evidence=n['evidence'],no_training_record=True));continue
        assert n['disposition'] in ('accept_edit','accept_decline')
        chosen=next(a for a in c['candidates'] if a['id']==n['candidate_id'])
        assert sha(ROOT/chosen['path'])==chosen['sha256']==n['candidate_sha256']
        inp=DATA/'inputs'/(c['id']+'.png');inp.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/c['image'],inp)
        if n['disposition']=='accept_edit':
            decision=validate_decision(chosen['decision']);target=DATA/'targets'/(c['id']+'.png')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/chosen['path'],target)
            replay=execute(load_rgb(inp),decision);assert np.array_equal(replay['result'],load_rgb(target))
            target_name=str(target.relative_to(ROOT));target_sha=sha(target)
        else:
            decision=validate_decision(dict(action='decline',reason='unreliable_quality'));target_name=target_sha=None
            assert execute(load_rgb(inp),decision)['result'] is None
        additions.append(dict(id=c['id'],source_id=next(g['prior_case_id'] for g in plan['groups'] if g['group_id']==c['group_id']),
            group_id=c['group_id'],domain=c['domain'],split='train',family='lighting',image=str(inp.relative_to(ROOT)),
            input_sha256=sha(inp),input_size=c['input_size'],instructions=c['instructions'],
            expected_change='Noticeably open shadows while preserving natural colours and recognizable detail; no obvious added grain or coloured blotches.',
            must_preserve='Subject/content/composition, recognizable fine structure, plausible colour, highlights; no visible grain or colour-mottle amplification.',
            failure_conditions=['Insufficient shadow lift','Visible added grain or coloured blotches','Lost fine structure, clipping, cast or unrequested change','Invalid output or executing a quality refusal'],
            instruction_origin='One AI-authored bilingual request template reused across matched TRAIN image conditions; synthetic underexposure/noise, not original GIER request or user feedback.',
            unique_new_instruction_template_pair=1,canonical_photo=c['original']['path'],original_source_sha256=c['original']['sha256'],
            source_partition='Original training pool; already part of240 R21 TRAIN original groups, new to quality-refusal supervision for this group.',
            condition_origin=c['condition'],action=decision['action'],decision=decision,parameters=decision.get('parameters'),target=target_name,target_sha256=target_sha,
            editor=EDITOR_VERSION,schema=SCHEMA_VERSION,status='direct_contact_and_native_AI_admitted_not_human_or_user_approval',
            label_evidence=n['evidence'],label_review=dict(candidate_manifest=str(manifest_path.relative_to(ROOT)),candidate_id=n['candidate_id'],
                triage_notes=str(triage_path.relative_to(ROOT)),native_notes=str(notes_path.relative_to(ROOT)),native_panel=n['panel']),
            finite_attempts_not_impossibility_proof=decision['action']=='decline'))
    cases=old+additions;assert len({c['id'] for c in cases})==len(cases)
    assert len({c['group_id'] for c in cases})==240
    records=[dict(id=c['id']+'-'+lang,case_id=c['id'],language=lang,domain=c['domain'],decision=c['decision'],instruction_origin=c['instruction_origin'],repeat=0)
        for c in cases for lang in ('zh','en')]
    write(DATA/'cases.json',cases);write(DATA/'new-cases.json',additions);write(DATA/'train-accepted.json',records)
    write(DATA/'excluded-candidates.json',excluded)
    quality=[c for c in cases if c['decision'].get('reason')=='unreliable_quality']
    counts=dict(total_semantic_condition_cases=len(cases),language_records=len(records),original_groups=240,
        portrait_groups=len({c['group_id'] for c in cases if c['domain']=='portrait'}),landscape_groups=len({c['group_id'] for c in cases if c['domain']=='landscape'}),
        actions=dict(collections.Counter(c['action'] for c in cases)),quality_refusal_cases=len(quality),quality_refusal_original_groups=len({c['group_id'] for c in quality}),
        new_cases=len(additions),new_edits=sum(c['action']=='edit' for c in additions),new_quality_refusals=sum(c['action']=='decline' for c in additions),
        new_original_photos=0,new_independent_original_groups=0,new_bilingual_instruction_templates=1,new_language_records=2*len(additions),
        excluded_uncertain=len(excluded),original_English_cases=154,AI_authored_cases=771+len(additions))
    write(DATA/'admission.json',dict(at=stamp(),status='AI_visual_admitted_not_user_approval',counts=counts,
        inherited=dict(path=str(source.relative_to(ROOT)),sha256=sha(source),cases=925,unchanged_json_objects=True),
        evidence={str(p.relative_to(ROOT)):sha(p) for p in (EXP/'quality-data-plan.json',manifest_path,native_path,triage_path,notes_path)},
        training_files={name:sha(DATA/name) for name in ('cases.json','new-cases.json','train-accepted.json','excluded-candidates.json')},
        GPU_hours=0,training_started=False,acceptance_opened=False,
        limitations=['Synthetic matched noise does not establish generalization to real camera noise.','Only one new bilingual instruction template; conditions/translations are not independent original photographs.',
            'Finite failed tool attempts support an unreliable-quality policy target, not mathematical impossibility.','Old925 labels retain prior evidence; not falsely re-reviewed this round.']))
    print(json.dumps(counts));print('New edit parameter counts:',dict(collections.Counter(json.dumps(c['parameters'],sort_keys=True) for c in additions if c['action']=='edit')))


if __name__=='__main__':main()

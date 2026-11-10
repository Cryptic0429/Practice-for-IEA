"""Admit explicit visual preferences, preserving every prior positive/input."""
import os
from pathlib import Path

from functional_v1.common import ROOT, read, write, sha, stamp

EXP = ROOT / 'experiments/functional-v18-20261107'
DATA = ROOT / 'data/training/functional-v18-20261107'
SOURCE = ROOT / 'data/training/functional-v15-20261106'


def streaming_sha(path):
    import hashlib
    value = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(1024*1024):
            value.update(block)
    return value.hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import json
    import numpy as np
    import torch
    from functional_v15.runtime import processor, build_batch
    from functional_v15.outcomes import execute, target_text, input_row
    from functional_v15.editor import load_rgb
    DATA.mkdir(exist_ok=False)
    selection = read(EXP/'preference-selection.json')
    assert len(selection['selected_case_ids']) == 19
    cases = read(SOURCE/'cases.json')
    by_id = {c['id']:c for c in cases}
    rows = read(SOURCE/'train-accepted.json')
    indices = {(r['case_id'],r['language']):i for i,r in enumerate(rows)}
    manifest = read(EXP/'review/manifest.json')
    bank = {c['case_id']:c for pair in manifest['items'] for c in pair['cases']}
    observations = [json.loads(s) for s in (EXP/'native-observations.jsonl').read_text().splitlines()]
    viewed = {c['case_id']:c for pair in observations for c in pair['cases']}
    cache = read(ROOT/'experiments/functional-v15-20261106/batch-cache.json')
    assert sha(SOURCE/'train-accepted.json') == cache['dataset_sha256']
    assert sha(SOURCE/'cases.json') == cache['cases_sha256']
    base_cache = ROOT/cache['cache_path']
    assert streaming_sha(base_cache) == cache['cache_sha256']
    batches = torch.load(base_cache,map_location='cpu',weights_only=True)
    assert len(batches)==len(rows)==1850
    p=processor()
    preferences=[]; negative_batches={}; checks=[]
    for cid in selection['selected_case_ids']:
        case=by_id[cid];item=bank[cid];note=viewed[cid]
        assert case['split']=='train'
        assert note['candidate_grades']['own_label']==dict(completion='completed',quality='acceptable')
        assert note['candidate_grades']['crossed_label'] in [dict(completion='partial',quality='acceptable'),dict(completion='completed',quality='minor_issue')]
        own=next(v for v in item['mapping'].values() if v['origin']=='own_label')
        crossed=next(v for v in item['mapping'].values() if v['origin']=='crossed_label')
        assert own['decision']==case['decision'] and own['decision']!=crossed['decision']
        assert own['decision']['parameters'].keys()==crossed['decision']['parameters'].keys()
        assert sha(ROOT/case['image'])==case['input_sha256']==item['input_sha256']
        rgb=load_rgb(ROOT/case['image'])
        for value in (own,crossed):
            assert sha(ROOT/value['path'])==value['sha256']
            assert np.array_equal(execute(rgb,value['decision'])['result'],load_rgb(ROOT/value['path']))
        for language in ('zh','en'):
            index=indices[cid,language];positive=batches[index]
            assert rows[index]['decision']==own['decision'] and rows[index]['id']==cid+'-'+language
            negative=build_batch(p,case,language,crossed['decision'])
            mask_positive=positive['labels'].eq(-100);mask_negative=negative['labels'].eq(-100)
            assert torch.equal(positive['input_ids'][mask_positive],negative['input_ids'][mask_negative])
            assert torch.equal(positive['pixel_values'],negative['pixel_values'])
            assert torch.equal(positive['image_grid_thw'],negative['image_grid_thw'])
            own_tokens=int(positive['labels'].ne(-100).sum())
            crossed_tokens=int(negative['labels'].ne(-100).sum())
            pid=cid+'-'+language
            negative_batches[pid]=dict(negative)
            preferences.append(dict(id=pid,case_id=cid,language=language,domain=case['domain'],family=case['family'],
                group_id=case['group_id'],chosen=own['decision'],rejected=crossed['decision'],
                chosen_image=own['path'],chosen_sha256=own['sha256'],rejected_image=crossed['path'],rejected_sha256=crossed['sha256'],
                base_row_index=index,instruction_origin=case['instruction_origin'],
                original_request=case['instructions'][language],native_observation=note))
            checks.append(dict(id=pid,assistant_tokens=own_tokens,crossed_assistant_tokens=crossed_tokens,
                exact_image_and_input_prefix_equal=True,input_keys=list(input_row(case,language)),
                chosen_text=target_text(own['decision']),crossed_text=target_text(crossed['decision'])))
    assert len(preferences)==len(negative_batches)==38
    pool=[dict(id=r['id'],kind='anchor',base_row_index=i) for i,r in enumerate(rows)]
    for repeat in range(4):
        pool.extend(dict(id='contrastive-'+r['id']+'-repeat'+str(repeat),kind='preference',
            preference_id=r['id'],case_id=r['case_id'],base_row_index=r['base_row_index'],repeat=repeat) for r in preferences)
    assert len(pool)==2002
    (DATA/'cases.json').write_bytes((SOURCE/'cases.json').read_bytes())
    (DATA/'train-accepted.json').write_bytes((SOURCE/'train-accepted.json').read_bytes())
    write(DATA/'preferences.json',preferences);write(DATA/'training-pool.json',pool)
    torch.save(negative_batches,EXP/'negative-batches.pt')
    groups={by_id[cid]['group_id']:by_id[cid]['domain'] for cid in selection['selected_case_ids']}
    write(EXP/'training-input-cache-audit.json',dict(at=stamp(),status='CPU_prefix_pixel_mask_and_exact_replay_checks_passed',
        base_cache_path=cache['cache_path'],base_cache_sha256=cache['cache_sha256'],base_cache_rows=1850,
        negative_cache_path=str((EXP/'negative-batches.pt').relative_to(ROOT)),negative_cache_sha256=streaming_sha(EXP/'negative-batches.pt'),
        negative_cache_rows=38,checks=checks,GPU_calls=0,model_forward_calls=0,
        unequal_assistant_lengths=sum(c['assistant_tokens']!=c['crossed_assistant_tokens'] for c in checks),
        ranking_score_policy='Mean assistant-token logp for each candidate, then fixed scale32. Raw sum-logp training is forbidden in this pilot; different numeric digit counts cannot be assumed equal length.',
        inference_input_policy='Exactly source image, one literal-language request, unchanged tool prompt. Candidate answers appear only in supervised assistant scoring, never in greedy input prefixes.'))
    write(DATA/'admission.json',dict(at=stamp(),status='explicit_visual_preferences_and_exact_prior_positives_admitted_before_training',
        unique_semantic_inputs=925,unique_language_inputs=1850,original_groups=240,source_domains={'portrait':120,'landscape':120},
        edit_keep_decline={'edit':780,'keep':80,'decline':65},unchanged_original_English_requests=154,AI_requests=771,
        new_photos=0,new_request_rewrites=0,new_positive_parameter_labels=0,
        preference_semantics=19,preference_language_records=38,preference_groups=len(groups),
        preference_group_domains={d:list(groups.values()).count(d) for d in ('portrait','landscape')},
        preference_semantic_domains={d:sum(by_id[cid]['domain']==d for cid in selection['selected_case_ids']) for d in ('portrait','landscape')},
        preference_repeat_per_epoch=4,anchor_rows_per_epoch=1850,preference_presentations_per_epoch=152,total_presentations_per_epoch=2002,
        source_dataset_sha256=sha(SOURCE/'train-accepted.json'),dataset_sha256=sha(DATA/'train-accepted.json'),cases_sha256=sha(DATA/'cases.json'),
        preferences_sha256=sha(DATA/'preferences.json'),training_pool_sha256=sha(DATA/'training-pool.json'),
        manual_selection_sha256=sha(EXP/'preference-selection.json'),native_observations_sha256=sha(EXP/'native-observations.jsonl'),
        initial_overview_observations_sha256=sha(EXP/'overview-observations.jsonl'),cache_audit_sha256=sha(EXP/'training-input-cache-audit.json'),
        training_launched=False,independent_user_feedback=0,
        limitation='Additional preference rows repeat existing inputs and positive labels. They are AI visual comparisons of synthetic mild/strong TRAIN conditions, not new real-camera captures or user feedback. Ambiguous alternatives are excluded; prior opinions retained.'))
    print('CPU-admitted unchanged1850 positives plus38 visual preferences (19 semantics);2002 presentations/epoch;no GPU.')


if __name__=='__main__':main()

"""Admit changed pixels only after direct review; preserve source and failed attempts."""
import copy
import hashlib
import json
import os
from collections import Counter, defaultdict
import numpy as np
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.editor import load_rgb
from .prepare_labels import EXP
from .outcomes import execute, input_row, system_prompt, validate_decision, EDITOR_VERSION, SCHEMA_VERSION

DATA = ROOT / 'data/training/functional-v11-20261106'


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not (DATA/'admission.json').exists()
    candidate_file = EXP/'candidate-cases.json'
    manifest_file = EXP/'label-candidates/manifest.json'
    m = read(manifest_file)
    assert m['all_cases_sha256'] == sha(candidate_file)
    assert m['editor_sha256'] == sha(ROOT/'scripts/functional_v11/editor.py')
    assert m['schema_sha256'] == sha(ROOT/'scripts/functional_v11/outcomes.py')
    raw = read(candidate_file)
    direct_file = EXP/'label-candidates/direct-observations.jsonl'
    notes = {n['id']: n for n in lines(direct_file)}
    expected = {c['id']: c for c in m['negative_curve_candidates'] + m['cool_cast_candidates']}
    assert len(notes) == len(expected) == 61 and set(notes) == set(expected)
    repair_file = EXP/'label-candidates/repairs/direct-observations.jsonl'
    repairs = {n['case_id']: n for n in lines(repair_file) if n['selected_replacement']}
    assert set(repairs) == {'native-train-n048', 'native-train-x250'}
    quality_file = EXP/'quality-revalidation/direct-observations.jsonl'
    quality_repairs = EXP/'quality-revalidation/repairs/direct-observations.jsonl'
    attempts = lines(quality_file) + lines(quality_repairs)
    assert len(attempts) == len({r['id'] for r in attempts}) == 53
    by_case = defaultdict(list)
    for r in attempts:
        assert r['quality'] != 'acceptable', 'A feasible successful edit takes precedence over refusal'
        for v in (r['input'], r['output']):
            assert sha(ROOT/v['path']) == v['sha256']
        by_case[r['case_id']].append(r)
    quality_ids = {c['id'] for c in m['inherited_quality_decisions_requiring_new_feasibility_review']}
    assert len(quality_ids) == 44 and set(by_case) == quality_ids
    prior = ROOT/'data/training/functional-v7-20261105'
    originals = read(prior/'cases.json')
    assert sha(prior/'cases.json') == '1948fbe5e40cdeae1d3e37fefedd4c87d9ebb7a2cf1eec3ff830229e8b98353a'
    allowed = {c['group_id']: c['domain'] for c in originals}
    assert len(allowed) == 160 and Counter(allowed.values()) == dict(portrait=80, landscape=80)
    inherited = {r['id']: r for r in m['unchanged_exact_replay']}
    cases, lineage, replay, audit, rows = [], [], [], [], []
    for c in copy.deepcopy(raw):
        cid = c['id']
        assert c['group_id'] in allowed and c['domain'] == allowed[c['group_id']] and c['split'] == 'train'
        assert sha(ROOT/c['image']) == c['input_sha256']
        if cid in expected:
            n = notes[cid]; e = expected[cid]
            assert n['input'] == e['input'] and n['candidate'] == e['candidate']
            assert n['manifest_sha256'] == sha(manifest_file)
            for v in (n['input'], n['candidate'], n['panel']):
                assert sha(ROOT/v['path']) == v['sha256']
            evidence = dict(kind='new_direct_AI_visual_review', review=str(direct_file.relative_to(ROOT)), observation=n)
            if cid in repairs:
                r = repairs[cid]
                assert r['completion'] == 'completed' and r['quality'] == 'acceptable'
                assert r['input'] == n['input'] and r['original_size_result_also_viewed']
                assert sha(ROOT/r['manifest']) == r['manifest_sha256']
                c.update(parameters=r['parameters'], decision=dict(action='edit', parameters=r['parameters']),
                         target=r['output']['path'], target_sha256=r['output']['sha256'])
                evidence.update(kind='new_direct_original_size_repair', selected_repair=r,
                                repair_review=str(repair_file.relative_to(ROOT)), failed_old_pixels_excluded=True)
            else:
                assert n['completion'] == 'completed' and n['quality'] == 'acceptable', cid
        else:
            old = c['previous_record']
            for path_key, hash_key in [('cases', 'cases_sha256'), ('admission', 'admission_sha256')]:
                assert sha(ROOT/old[path_key]) == old[hash_key]
            original = next(x for x in read(ROOT/old['cases']) if x['id'] == cid)
            assert all(c[k] == original[k] for k in ('image', 'input_sha256', 'instructions', 'decision', 'target', 'target_sha256'))
            evidence = dict(kind='exact_prior_AI_evidence_inherited_after_pixel_replay', previous_record=old,
                            limitation='Same request and exact pixels; not a new independent judgment. Previous unresolved uncertainty is preserved.')
            assert cid in inherited
            if cid in quality_ids:
                assert c['decision'] == dict(action='decline', reason='unreliable_quality')
                assert all(r['input']['sha256'] == c['input_sha256'] for r in by_case[cid])
                evidence.update(kind='quality_decline_revalidated_after_tool_change',
                    feasibility_attempt_ids=[r['id'] for r in by_case[cid]],
                    review=str(quality_file.relative_to(ROOT)), repair_review=str(quality_repairs.relative_to(ROOT)),
                    conclusion='A reliable result satisfying the request was not found among the directly viewed attempts with these global tools.',
                    limitation='Finite parameter attempts and AI review; not proof that every possible setting is impossible. Synthetic corruption is explicitly distinct from real camera noise.')
        decision = validate_decision(c['decision']); c['parameters'] = decision.get('parameters', {} if decision['action']=='keep' else None)
        c.update(action=decision['action'], editor=EDITOR_VERSION, schema=SCHEMA_VERSION,
                 status='direct_AI_admitted_not_human_or_user_approval', label_evidence=evidence)
        result = execute(load_rgb(ROOT/c['image']), decision)
        if c['action'] == 'decline':
            assert result['result'] is None and c['target'] is None
        else:
            assert sha(ROOT/c['target']) == c['target_sha256']
            assert np.array_equal(result['result'], load_rgb(ROOT/c['target'])), cid
            if c['action'] == 'keep':
                assert np.array_equal(result['result'], load_rgb(ROOT/c['image']))
        replay.append(dict(case_id=cid, action=c['action'], input_sha256=c['input_sha256'], target_sha256=c['target_sha256'], exact_replay=True))
        cases.append(c); lineage.append(dict(case_id=cid, action=c['action'], **evidence))
        for lang in ('zh', 'en'):
            row = input_row(c, lang)
            poisoned = dict(c, decision={'action':'keep'}, target='FORBIDDEN_REFERENCE', parameters={'sepia':99},
                expected_change='FORBIDDEN_CRITERIA', label_evidence='FORBIDDEN_REVIEW', condition='FORBIDDEN_CONDITION',
                family='FORBIDDEN_FAMILY', synthetic_corruption='FORBIDDEN_NOISE', previous_record='FORBIDDEN_HISTORY')
            assert input_row(poisoned, lang) == row
            assert row['conversations'] == [dict(role='system', content=system_prompt()), dict(role='user', content='<image>\n'+c['instructions'][lang])]
            audit.append(dict(case_id=cid, language=lang, metadata_poisoning_unchanged=True,
                input_hash=hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()))
            rows.append(dict(id=cid+'-'+lang, case_id=cid, language=lang, domain=c['domain'], decision=decision,
                             instruction_origin=c['instruction_origin'], repeat=0))
    assert len(cases) == len({c['id'] for c in cases}) == 672
    assert len(rows) == len({r['id'] for r in rows}) == 1344
    assert Counter(c['action'] for c in cases) == dict(edit=548, keep=60, decline=64)
    native = sum(c['instruction_origin'].startswith('English untouched original GIER') for c in cases)
    assert native == 97
    isolation = []
    for name, item in read(ROOT/'experiments/functional-v10-20261106/evaluation-protocol.json')['case_files'].items():
        path=ROOT/item['path']; assert sha(path) == item['sha256']
        assert not set(allowed) & {c['group_id'] for c in read(path)}, name
        isolation.append(dict(partition=name, manifest=item, training_groups_disjoint=True,
            scope='Metadata/group hashes only. Inherits previous original-source near-duplicate exclusion; no heldout images or predictions opened.'))
    provenance_files = [candidate_file, manifest_file, direct_file, repair_file, quality_file, quality_repairs,
        prior/'cases.json', prior/'admission.json', ROOT/'data/training/functional-v10-20261106/cases.json',
        ROOT/'data/training/functional-v10-20261106/admission.json']
    provenance_files += list((EXP/'label-candidates/repairs').glob('*/manifest.json'))
    provenance_files += list((EXP/'quality-revalidation').glob('**/manifest.json'))
    write(DATA/'cases.json', sorted(cases, key=lambda c:c['id']))
    write(DATA/'train-accepted.json', rows); write(DATA/'label-lineage.json', lineage)
    write(EXP/'target-replay.json', dict(at=stamp(), status='passed', records=replay))
    write(EXP/'input-audit.json', dict(at=stamp(), status='passed', records=audit,
          only_input_image_one_request_and_tool_prompt=True))
    write(EXP/'source-isolation.json', dict(at=stamp(), status='passed', records=isolation,
          groups=160, all_within_prior_training_pool=True, original_photo_domains=dict(Counter(allowed.values()))))
    write(DATA/'admission.json', dict(at=stamp(), status='direct_AI_visual_admitted_not_human_or_user_approval',
        semantic_cases=672, unique_language_rows=1344, actual_rows_each_route=1344, source_groups=160,
        photo_domains=dict(Counter(allowed.values())), case_domain_counts=dict(Counter(c['domain'] for c in cases)),
        inherited_unchanged_edit_keep_and_unsupported=567, changed_negative_labels=51,
        changed_labels_repaired=2, new_synthetic_cool_cast_cases=10, quality_declines_revalidated=44,
        new_photograph_groups_added=0, outcome_counts=dict(Counter(c['action'] for c in cases)),
        original_GIER_English_cases=97, original_English_with_AI_Chinese_translation_rows=194,
        AI_authored_cases=575, AI_authored_rows=1150, original_English_semantic_fraction=97/672,
        family_counts=dict(Counter(c['family'] for c in cases)),
        training_data_sha256=sha(DATA/'train-accepted.json'), cases_sha256=sha(DATA/'cases.json'), lineage_sha256=sha(DATA/'label-lineage.json'),
        provenance_hashes={str(p.relative_to(ROOT)):sha(p) for p in provenance_files},
        selection='Visible completion plus basic quality, never L1 or reward score. Failed attempts preserved and excluded.',
        frozen_Qwen_screening='Earlier critic calibration failed; direct Codex AI visual review is final, not human approval.',
        limitations=['160 original groups were already used by the parent; added conditions are not independent photographs.',
            '567 records inherit unchanged prior evidence explicitly; all61 changed/new targets reviewed,2 replaced after failure.',
            '44 quality declines rechecked with53 actual attempts; failure of finite attempts is not a universal impossibility proof.',
            'Artificial warm cast, exposure and block noise are documented synthetic conditions, not real-camera generalization evidence.',
            'No trial upload or user feedback enters training; independent user feedback remains zero.'], no_GPU_training_started=True))
    print('Admitted672 semantics/1344 bilingual,160groups80/80;548edit60keep64decline;97 original-English575AI. Audits passed.')


if __name__ == '__main__':
    main()

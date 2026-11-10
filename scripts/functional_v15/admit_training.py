"""Materialize only reviewed shadow repairs; inherit exact unchanged R13 evidence."""
import copy, hashlib, json, os
from collections import Counter
import numpy as np
from functional_v1.common import ROOT, read, write, sha, stamp
from .editor import load_rgb
from .outcomes import execute, input_row, validate_decision, EDITOR_VERSION, SCHEMA_VERSION

EXP = ROOT / 'experiments/functional-v15-20261106'
DATA = ROOT / 'data/training/functional-v15-20261106'
SOURCE = ROOT / 'data/training/functional-v13-20261106'
SOURCE_SHA = '4d24be767daeb2ae523ab0988c7d2887d650a54e9f2de263e1b61211b832a39b'
CRITERIA_SHA = 'f6331f5c7de23553828acce63a860b8cbb311d206f2dbe2179ce01df19619fb6'


def notes(name):
    rows = [json.loads(line) for line in (EXP / name).read_text().splitlines()]
    assert len(rows) == len({r['id'] for r in rows})
    return {r['id']: r for r in rows}


def verify_native(note):
    parents = {str((ROOT / i['path']).parent) for i in note['images']}
    assert len(parents) == 1
    from pathlib import Path
    assert sha(Path(next(iter(parents))) / 'manifest.json') == note['manifest_sha256']
    for i in note['images']:
        assert sha(ROOT / i['path']) == i['sha256']
    for i in note.get('paired_native', []):
        assert sha(ROOT / i['path']) == i['sha256']


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not DATA.exists(), 'Preserve any prior materialization; audit before replacing it'
    assert sha(SOURCE / 'cases.json') == SOURCE_SHA
    criteria_file = EXP / 'functional-review-criteria.json'
    assert sha(criteria_file) == CRITERIA_SHA
    criteria = {c['id']: c for c in read(criteria_file)['items']}
    candidate_manifest = EXP / 'label-candidates/manifest.json'
    candidates = read(candidate_manifest)
    assert candidates['source_cases_sha256'] == SOURCE_SHA
    bank = {i['id']: i for i in candidates['items']}
    raw = notes('label-observations.jsonl'); native = notes('label-native-observations.jsonl')
    assert set(raw) == set(bank) and len(bank) == 132 and len(native) == 116
    for cid, note in raw.items():
        item = bank[cid]
        assert note['manifest_sha256'] == sha(candidate_manifest)
        assert note['criteria_sha256'] == CRITERIA_SHA
        assert sha(ROOT / item['panel']['path']) == note['panel']['sha256'] == item['panel']['sha256']
        assert note['candidate_image_sha256'] == sha(ROOT / item['candidate']['path']) == item['candidate']['sha256']
        if cid in native:
            n = native[cid]; verify_native(n)
            assert n['candidate_sha256'] == item['candidate']['sha256'] and n['input_sha256'] == item['input']['sha256']
        assert note['quality'] != 'U' or cid in native
    selected_file = EXP / 'repair-selection.json'
    selection = read(selected_file)
    assert selection['observations_sha256'] == sha(EXP / 'repair-observations.jsonl')
    repairs = {r['id']: r for r in selection['items']}
    assert len(repairs) == 11
    for cid, selected in repairs.items():
        n = selected['native_observation']
        assert n['stage'] == 'native' and n['completion'] == 'C' and n['quality'] == 'A'
        source = EXP / 'label-repairs' / selected['bundle'] / 'manifest.json'
        assert sha(source) == n['manifest_sha256']
        item = next(i for i in read(source)['items'] if i['id'] == cid)
        result = next(r for r in item['results'] if r['candidate'] == selected['candidate'])
        assert result == selected['selected_result'] == n['result']
        assert item['instructions'] == criteria[cid]['original_requests_unchanged']
        assert sha(ROOT / result['image']) == result['image_sha256']
        assert all(sha(ROOT / i['path']) == i['sha256'] for i in n['regions'])
        native_manifest = EXP / 'native-review/repairs' / selected['bundle'] / 'manifest.json'
        asset = read(native_manifest)
        assert asset['source_manifest_sha256'] == sha(source)
        regions = [i for i in next(i for i in asset['items'] if i['id'] == cid)['images']
                   if i['role'] in ('INPUT', selected['candidate'])]
        assert regions == n['regions']
    failed = {cid for cid in raw if (native.get(cid, raw[cid])['completion'], native.get(cid, raw[cid])['quality']) != ('C', 'A')}
    assert failed == set(repairs), 'A failed or unresolved label must not silently enter training'

    quality_manifest = EXP / 'quality-challenges/manifest.json'
    challenges = {i['id']: i for i in read(quality_manifest)['items']}
    qraw = notes('quality-challenge-observations.jsonl'); qnative = notes('quality-native-observations.jsonl')
    assert set(qraw) == set(challenges) and len(qraw) == 44 and len(qnative) == 3
    for cid, item in challenges.items():
        n = qraw[cid]
        assert n['manifest_sha256'] == sha(quality_manifest) and n['criteria_sha256'] == CRITERIA_SHA
        assert sha(ROOT / item['panel']['path']) == n['panel']['sha256'] == item['panel']['sha256']
        assert sha(ROOT / item['old']['path']) == item['old']['sha256']
        assert sha(ROOT / item['candidate']['path']) == n['candidate_image_sha256'] == item['candidate']['sha256']
        if cid in qnative:
            verify_native(qnative[cid])
            assert qnative[cid]['candidate_sha256'] == item['candidate']['sha256']
        final = qnative.get(cid, n)
        assert final['quality'] == 'm' and final['completion'] == 'C'
        actual = execute(load_rgb(ROOT / item['input']['path']), dict(action='edit', parameters=item['parameters']))
        assert np.array_equal(actual['result'], load_rgb(ROOT / item['candidate']['path']))

    cases = copy.deepcopy(read(SOURCE / 'cases.json')); assert len(cases) == 925
    quality_declines = {c['id'] for c in cases if c['decision'] == dict(action='decline', reason='unreliable_quality')}
    assert quality_declines == set(challenges)
    replay = []; audit = []; rows = []
    for c in cases:
        cid = c['id']; requirement = criteria[cid]
        assert c['instructions'] == requirement['original_requests_unchanged']
        assert c['input_sha256'] == requirement['input_sha256'] == sha(ROOT / c['image'])
        c.update(expected_change=requirement['required_change'], must_preserve=requirement['must_preserve'],
                 failure_conditions=requirement['failure_conditions'], review_criteria_origin=requirement['criteria_origin'])
        c['r15_evidence'] = dict(kind='exact_R13_input_request_and_prior_evidence_inheritance',
            source_cases=str((SOURCE / 'cases.json').relative_to(ROOT)), source_cases_sha256=SOURCE_SHA,
            source_admission_sha256=sha(SOURCE / 'admission.json'), criteria_sha256=CRITERIA_SHA)
        if cid in bank:
            if cid in repairs:
                result = repairs[cid]['selected_result']
                c.update(decision=result['decision'], parameters=result['decision']['parameters'],
                         target=result['image'], target_sha256=result['image_sha256'])
                proof = dict(kind='direct_AI_repaired_label_not_human_approval', selection=repairs[cid])
            else:
                item = bank[cid]
                c.update(target=item['candidate']['path'], target_sha256=item['candidate']['sha256'])
                proof = dict(kind='direct_AI_revalidated_shadow_label_not_human_approval')
            proof.update(raw=raw[cid], native=native.get(cid), source_label_evidence=c['label_evidence'])
            c['label_evidence'] = proof
            c['r15_evidence']['changed_target_direct_review'] = True
        if cid in challenges:
            c['r15_evidence']['quality_feasibility_revalidation'] = dict(raw=qraw[cid], native=qnative.get(cid),
                challenge=challenges[cid], scope='No reliable edit found in the inherited finite bank plus this new-tool challenge. '
                    'Retained existing conservative refusal; not a claim that every possible parameter combination fails.')
        c.update(editor=EDITOR_VERSION, schema=SCHEMA_VERSION, status='direct_or_exact_inherited_AI_admitted_not_human_or_user_approval')
        decision = validate_decision(c['decision']); assert c['action'] == decision['action']
        actual = execute(load_rgb(ROOT / c['image']), decision)
        if decision['action'] == 'decline':
            assert actual['result'] is None and c['target'] is None and c['target_sha256'] is None
        else:
            assert sha(ROOT / c['target']) == c['target_sha256']
            assert np.array_equal(actual['result'], load_rgb(ROOT / c['target'])), cid
            if decision['action'] == 'keep': assert np.array_equal(actual['result'], load_rgb(ROOT / c['image']))
        replay.append(dict(id=cid, input_sha256=c['input_sha256'], target_sha256=c['target_sha256'], exact_replay=True))
        for lang in ('zh', 'en'):
            row = input_row(c, lang)
            poison = dict(c, target='FORBIDDEN_TARGET', parameters='FORBIDDEN_PARAMETERS', decision={'action':'keep'},
                label_evidence='FORBIDDEN_REVIEW', r15_evidence='FORBIDDEN_REVIEW', expected_change='FORBIDDEN_CRITERIA',
                must_preserve='FORBIDDEN_CRITERIA', failure_conditions='FORBIDDEN_CRITERIA', family='FORBIDDEN_FAMILY',
                native_annotation='FORBIDDEN_ANNOTATION', review_criteria_origin='FORBIDDEN_ORIGIN')
            assert input_row(poison, lang) == row
            audit.append(dict(case_id=cid, language=lang, metadata_poisoning_unchanged=True,
                input_sha256=hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()))
            rows.append(dict(id=cid+'-'+lang, case_id=cid, language=lang, domain=c['domain'], decision=decision,
                             instruction_origin=c['instruction_origin'], repeat=0))
    source_groups = {c['group_id']: c['domain'] for c in cases}
    assert len(source_groups) == 240 and Counter(source_groups.values()) == dict(portrait=120, landscape=120)
    isolation = ROOT / 'experiments/functional-v13-20261106/source-isolation-extended.json'
    assert read(isolation)['flags'] == []
    protocol = ROOT / 'experiments/functional-v13-20261106/evaluation-protocol.json'
    for name, info in read(protocol)['case_files'].items():
        assert sha(ROOT / info['path']) == info['sha256']
        assert not set(source_groups) & {c['group_id'] for c in read(ROOT / info['path'])}, name
    write(DATA / 'cases.json', sorted(cases, key=lambda c:c['id']))
    write(DATA / 'train-accepted.json', rows)
    write(DATA / 'failed-proposals.json', dict(initial_failed_labels=[dict(id=cid, raw=raw[cid], native=native.get(cid)) for cid in sorted(failed)],
        all_repair_observations=[json.loads(line) for line in (EXP / 'repair-observations.jsonl').read_text().splitlines()],
        policy='All failures and unselected proposals preserved; quality-U is not accepted.'))
    write(EXP / 'target-replay.json', dict(at=stamp(), status='passed', records=replay, quality_challenge_replays=44, GPU_hours=0))
    write(EXP / 'input-audit.json', dict(at=stamp(), status='passed', records=audit, only_image_literal_request_tools=True, GPU_hours=0))
    write(EXP / 'source-isolation-inheritance.json', dict(status='exact_source_inputs_groups_inherited_not_new_fingerprint_scan',
        source_cases_sha256=SOURCE_SHA, inherited_audit=str(isolation.relative_to(ROOT)), inherited_audit_sha256=sha(isolation),
        group_disjointness_rechecked=True, new_photographs=0, synthetic_conditions_added=0,
        limitations='Prior fingerprint/metadata checks are finite; related captures can still escape detection.'))
    native_origin = lambda c:c['instruction_origin']=='original_GIER_English_AI_Chinese' or c['instruction_origin'].startswith('English untouched original GIER')
    sources = [candidate_manifest, quality_manifest, selected_file, criteria_file] + list(EXP.glob('*observations.jsonl'))
    write(DATA / 'admission.json', dict(at=stamp(), status='direct_or_exact_inherited_AI_labels_admitted_not_human_or_user_approval',
        semantic_cases=len(cases), unique_language_rows=len(rows), source_groups=len(source_groups), photo_domains=dict(Counter(source_groups.values())),
        outcomes=dict(Counter(c['action'] for c in cases)), original_English_cases=sum(map(native_origin,cases)),
        AI_authored_cases=sum(not native_origin(c) for c in cases), new_photographs=0, instruction_rewrites=0,
        unchanged_R13_replays=793, changed_shadow_targets=132, reviewed_repairs=11, executed_repair_proposals=24,
        quality_refusals_revalidated=44, quality_refusal_scope='Existing conservative finite-bank reliability labels, not universal impossibility claims.',
        training_data_sha256=sha(DATA/'train-accepted.json'), cases_sha256=sha(DATA/'cases.json'),
        evidence_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},
        frozen_Qwen_screening='Not used as admission judge because frozen critic calibration failed in R8; explicit deviation from initial plan.',
        limitations=['AI visual review only; no independent human or user approval.',
            '793 unchanged labels inherit exact old evidence and limitations, not 793 new image reviews.',
            '925 cases are 240 underlying photo groups, with explicit AI/synthetic conditions; translations are not independent samples.',
            'Large-image native QA covers listed critical regions, not every pixel.',
            'No reliability or usefulness claim about unknown-photo model predictions follows from label acceptance.'],
        no_trial_upload_or_feedback_in_training=True, no_GPU_backprop_started=True))
    print(json.dumps({k:v for k,v in read(DATA/'admission.json').items() if k not in ('evidence_hashes','limitations')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

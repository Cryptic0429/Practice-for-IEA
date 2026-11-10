"""Admit executed, reviewed labels with an explicit edit/keep/decline decision.

Selection is based on persisted direct observations, never a pixel-distance
score. Replaced and excluded requests retain their original failure records.
"""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.editor import load_rgb
from .diagnose_editor import EXP
from .outcomes import input_row, system_prompt, validate_decision, execute, EDITOR_VERSION, SCHEMA_VERSION


DATA = ROOT / 'data/training/functional-v7-20261105'


def good(note):
    return note['completion'] == 'completed' and note['quality'] == 'acceptable'


def main():
    assert not (DATA / 'admission.json').exists(), 'Preserve the admitted dataset'
    previous = ROOT / 'data/training/functional-v6-20261105'
    originals = read(previous / 'cases.json')
    assert sha(previous / 'cases.json') == '375c98f1f53730c0e926677e90bb18cdf630bcd00730aa4223f631be10c045a3'
    prior = read(previous / 'admission.json')
    assert prior['semantic_cases'] == 460 and prior['source_groups'] == 160
    by_id = {c['id']: c for c in originals}
    folder = EXP / 'soft-label-candidates'
    manifest = read(folder / 'manifest.json')
    notes = {(n['case_id'], n['label']): n for n in read(folder / 'direct-label-observations.json')['observations']}
    assert len(notes) == 328 and len(manifest['sheets']) == 55
    full = {n['case_id']: n for n in read(folder / 'full-resolution-observations.json')['observations']}
    assert len(full) == 30
    new_notes = {}
    for sheet in manifest['sheets']:
        assert sha(ROOT / sheet['sheet']) == sheet['sha256']
        for row in sheet['entries']:
            label = next(v['label'] for v in row['variants'] if v['role'] == 'new')
            new_notes[row['case_id']] = notes[(row['case_id'], label)]
    cases = {}
    provenance = []
    sources = []

    def add(raw, decision, target, target_hash, evidence, source_id=None):
        c = copy.deepcopy(raw)
        assert c['id'] not in cases and c['split'] == 'train'
        c.update(previous_action=c.get('action'), action=decision['action'], decision=validate_decision(decision),
                 parameters=decision.get('parameters', {} if decision['action'] == 'keep' else None),
                 target=target, target_sha256=target_hash, editor=EDITOR_VERSION, schema=SCHEMA_VERSION,
                 status='direct_AI_admitted_not_human_or_user_approval', label_evidence=evidence)
        cases[c['id']] = c
        provenance.append(dict(case_id=c['id'], source_case_id=source_id or c['id'], **evidence))

    for candidate in manifest['cases']:
        cid = candidate['case_id']
        raw = by_id[cid]
        assert candidate['input_image'] == raw['image'] and candidate['instructions'] == raw['instructions']
        if candidate['pixel_identical_to_previously_reviewed_label']:
            assert np.array_equal(load_rgb(ROOT / candidate['new_candidate']), load_rgb(ROOT / raw['target']))
            note = dict(kind='exact_prior_reviewed_pixels', previous_admission=str((previous / 'admission.json').relative_to(ROOT)),
                        previous_admission_sha256=sha(previous / 'admission.json'))
        else:
            review = new_notes[cid]
            if cid in full:
                review = full[cid]
                assert review['image_sha256'] == candidate['new_candidate_sha256']
            if not good(review):
                continue
            note = dict(kind='new_direct_visual_review', review_file=str((folder / ('full-resolution-observations.json' if cid in full else 'direct-label-observations.json')).relative_to(ROOT)),
                        review_evidence=review['evidence'])
        add(raw, candidate['candidate_decision'], candidate['new_candidate'], candidate['new_candidate_sha256'], note)

    # These choices were made after looking at the result images, not from their
    # numeric parameters. Other candidates and all unsuccessful proposals stay.
    choices = {
        'label-repairs': {
            'train-045-restore-exposure-strong': '5e3bcb19f59b',
            'train-043-restore-exposure-mild': '67dfe82ce52f',
            'train-043-restore-exposure-strong': 'ab341193fb4e',
            'train-114-restore-exposure-strong': 'cd71bf7fe562',
            'train-191-restore-exposure-strong': 'd14ec102753a',
            'native-train-n180': '67aa3238a3bd',
            'native-train-x035': '8b3a8d91db39'},
        'label-quality-repairs': {
            'native-train-n119': 'be66ad10b85a',
            'native-train-x030': 'fe3620704f35',
            'native-train-n029': '20b76a99f3ca',
            'native-train-n068': '7c1c066328a8'},
    }
    for name, selected in choices.items():
        f = EXP / name
        m = read(f / 'manifest.json')
        panel_notes = {(n['case_id'], n['label']): n for n in read(f / 'direct-observations.json')['observations']}
        detail = {(n['case_id'], n['label']): n for n in read(f / 'full-resolution-observations.json')['observations']}
        for cid, label in selected.items():
            c = next(c for c in m['records'] if c['case_id'] == cid and c.get('label') == label)
            n = detail.get((cid, label), panel_notes[(cid, label)])
            assert good(n) and c['instructions'] == by_id[cid]['instructions'] and c['input_image'] == by_id[cid]['image']
            add(by_id[cid], dict(action='edit', parameters=c['parameters']), c['image'], c['image_sha256'],
                dict(kind='same_input_request_directly_reviewed_label_repair', review_file=str((f / ('full-resolution-observations.json' if (cid, label) in detail else 'direct-observations.json')).relative_to(ROOT)),
                     label=label, review_evidence=n['evidence']))

    decisions = EXP / 'decision-candidates'
    m = read(decisions / 'manifest.json')
    reviewed = {n['case_id']: n for n in read(decisions / 'direct-observations.json')['observations']}
    detail = {n['case_id']: n for n in read(decisions / 'full-resolution-observations.json')['observations']}
    assert len(reviewed) == len(m['records']) == 30
    for c in m['records']:
        n = detail.get(c['id'], reviewed[c['id']])
        if c['action'] == 'edit' and not good(n):
            continue
        if c['action'] == 'decline':
            assert n['completion'] == 'appropriate_decline' and n['quality'] == 'not_applicable'
            assert c['target'] is None and c['parameters'] is None
        add(c, c['decision'], c.get('target'), c.get('target_sha256'),
            dict(kind=c['kind'], review_file=str((decisions / ('full-resolution-observations.json' if c['id'] in detail else 'direct-observations.json')).relative_to(ROOT)),
                 review_evidence=n['evidence']))

    brown = EXP / 'decision-brown-repair'
    c = read(brown / 'manifest.json')['candidate']
    n = read(brown / 'direct-observation.json')
    assert good(n) and n['image_sha256'] == c['target_sha256']
    add(c, c['decision'], c['target'], c['target_sha256'],
        dict(kind='reviewed_repair_of_new_AI_counterrequest', review_file=str((brown / 'direct-observation.json').relative_to(ROOT)), review_evidence=n['evidence']),
        source_id='native-train-n013-AI-brown-v7')

    # Same 160 original photograph groups: preserve the existing source-isolation
    # audit without opening a heldout image or prediction in this admission.
    original_groups = {c['group_id']: c['domain'] for c in originals}
    assert {c['group_id'] for c in cases.values()} == set(original_groups)
    assert Counter(original_groups.values()) == dict(portrait=80, landscape=80)
    known_inputs = {(c['group_id'], c['image'], c['input_sha256']) for c in originals}
    for c in originals:
        if c.get('canonical_photo'):
            known_inputs.add((c['group_id'], c['canonical_photo'], sha(ROOT / c['canonical_photo'])))
    replay = []
    audit = []
    rows = []
    conflict_source = read(EXP / 'soft-contrast-capacity/baseline-review-consistency.json')['conflicts']
    capacity = read(EXP / 'soft-contrast-capacity/manifest.json')['records']
    conflict_pixels = [(n['previous']['case_id'], load_rgb(ROOT / next(r['image'] for r in capacity if r['image_sha256'] == n['image_sha256']))) for n in conflict_source]
    for c in cases.values():
        assert (c['group_id'], c['image'], c['input_sha256']) in known_inputs
        assert sha(ROOT / c['image']) == c['input_sha256']
        rgb = load_rgb(ROOT / c['image'])
        result = execute(rgb, c['decision'])
        if c['action'] == 'decline':
            assert result['result'] is None and result['status'] == 'declined' and c['target'] is None
        else:
            assert sha(ROOT / c['target']) == c['target_sha256']
            target = load_rgb(ROOT / c['target'])
            assert np.array_equal(result['result'], target)
            assert not any(cid == c['id'] and np.array_equal(pixels, target) for cid, pixels in conflict_pixels), 'Same-pixel review dispute cannot be silently admitted'
            if c['action'] == 'keep':
                assert c['family'] == 'identity' and np.array_equal(rgb, target) and result['status'] == 'unchanged'
        replay.append(dict(case_id=c['id'], action=c['action'], status=result['status'], input_sha256=c['input_sha256'],
                           target_sha256=c['target_sha256'], exact_target_replay=c['action'] != 'decline',
                           no_result_for_decline=c['action'] == 'decline'))
        for language in ('zh', 'en'):
            row = input_row(c, language)
            poisoned = dict(c, parameters={'sepia': 99}, decision={'action': 'keep'}, target='FORBIDDEN_LABEL',
                            expected_change='FORBIDDEN_REVIEW', reference='FORBIDDEN_REFERENCE', family='FORBIDDEN_FAMILY',
                            label_evidence={'FORBIDDEN': True}, target_statistics='FORBIDDEN_STATS')
            assert input_row(poisoned, language) == row
            assert row['conversations'] == [dict(role='system', content=system_prompt()), dict(role='user', content='<image>\n' + c['instructions'][language])]
            audit.append(dict(case_id=c['id'], language=language, metadata_poisoning_unchanged=True,
                              input_sha256=hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()))
            rows.append(dict(id=c['id'] + '-' + language, case_id=c['id'], language=language, domain=c['domain'],
                             decision=c['decision'], instruction_origin=c['instruction_origin'], repeat=0))
    assert len(rows) == len({r['id'] for r in rows}) == len(cases) * 2
    ordered = sorted(cases.values(), key=lambda c: c['id'])
    excluded = [dict(case_id=c['id'], family=c['family'], group_id=c['group_id'],
                     reason='No completed/acceptable target admitted for the original task; any new AI counterrequest is separate.')
                for c in originals if c['id'] not in cases]
    write(DATA / 'cases.json', ordered)
    write(DATA / 'train-accepted.json', rows)
    write(DATA / 'label-lineage.json', provenance)
    write(DATA / 'excluded-original-tasks.json', excluded)
    write(EXP / 'target-replay.json', dict(at=stamp(), status='passed', records=replay))
    write(EXP / 'input-audit.json', dict(at=stamp(), status='passed', records=audit,
                                       only_input_image_one_request_and_tool_prompt=True, hidden_metadata_invariant=True))
    original_native = [c for c in ordered if c['instruction_origin'].startswith('English untouched original GIER')]
    outcome_counts = dict(Counter(c['action'] for c in ordered))
    metadata_files = [previous / 'admission.json', previous / 'cases.json', folder / 'manifest.json',
                      folder / 'direct-label-observations.json', folder / 'full-resolution-observations.json',
                      EXP / 'label-repairs/direct-observations.json', EXP / 'label-repairs/full-resolution-observations.json',
                      EXP / 'label-quality-repairs/direct-observations.json', EXP / 'label-quality-repairs/full-resolution-observations.json',
                      decisions / 'direct-observations.json', decisions / 'full-resolution-observations.json', brown / 'direct-observation.json']
    write(DATA / 'admission.json', dict(
        at=stamp(), status='direct_AI_visual_admitted_not_human_or_user_approval', semantic_cases=len(ordered),
        unique_language_rows=len(rows), actual_rows_each_route=len(rows), source_groups=len(original_groups),
        photo_domains=dict(Counter(original_groups.values())), outcome_counts=outcome_counts,
        original_GIER_English_cases=len(original_native), original_English_with_AI_Chinese_translation_rows=len(original_native) * 2,
        AI_authored_cases=len(ordered) - len(original_native), instruction_origins=dict(Counter(c['instruction_origin'] for c in ordered)),
        task_families=dict(Counter(c['family'] for c in ordered)), original_task_exclusions=len(excluded),
        source_isolation='Exactly the same 160 original photograph groups and only original input hashes or their previously registered canonical PNGs; inherits the round6 isolation audit. No new heldout data opened.',
        oversampling=False, cases_sha256=sha(DATA / 'cases.json'), dataset_sha256=sha(DATA / 'train-accepted.json'),
        lineage_sha256=sha(DATA / 'label-lineage.json'), evidence_files={str(p.relative_to(ROOT)): sha(p) for p in metadata_files},
        editor=EDITOR_VERSION, schema=SCHEMA_VERSION, GPU_hours=0, new_training_started=False,
        limitations='All judgments are AI-assisted, with Codex direct visual verification. No human approval or actual user usability feedback. Frozen original Qwen visual screening previously failed directional calibration and is not treated as a second reliable reviewer. New AI counterrequests never count as original-request success. Quality declines reflect current tool reliability, not universal image impossibility.'))
    print(json.dumps(dict(cases=len(ordered), groups=len(original_groups), rows=len(rows), outcomes=outcome_counts,
                          original_GIER_cases=len(original_native), excluded_original_tasks=len(excluded)), ensure_ascii=False))


if __name__ == '__main__':
    main()

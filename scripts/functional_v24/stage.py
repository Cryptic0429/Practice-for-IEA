"""Stage reviewed GIER originals and parameter-label candidates for a later audit."""
import shutil

import numpy as np

from functional_v1.common import ROOT, read, sha, stamp, write
from functional_v15.editor import load_rgb
from functional_v15.outcomes import execute, target_text
from .probe import EXP, protected_sources

DATA = ROOT / 'data/training/functional-v24-20261107'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    plan = read(EXP/'plan.json')
    iso = read(EXP/'isolation.json')
    review = read(EXP/'visual-review.json')
    scan_path = EXP/'scan.json'
    require(sha(scan_path) == plan['scan_sha256'], 'Source scan has changed')
    sources = {c['id']: c for c in read(scan_path)['rows']}
    require(sha(EXP/'plan.json') == iso['plan_sha256'] and not iso['flags'], 'Source isolation failed')
    protected = protected_sources()
    require(all(protected.get(p) in ('GIER_val', 'GIER_test') for p in iso['missing']),
            'Missing a protected internal source')
    runs = {}
    for name, prefix, plan_name, digest_key in (
        ('initial', EXP, 'plan.json', 'initial_replay_sha256'),
        ('refined', EXP/'refinements', 'refinement-plan.json', 'refined_replay_sha256'),
    ):
        manifest = prefix/'replay.json'
        require(sha(manifest) == review[digest_key], 'Reviewed replay has changed')
        replay = read(manifest)
        require(replay['plan_sha256'] == sha(EXP/plan_name), 'Replay plan has changed')
        for case in replay['cases']:
            require(iso['candidate_source_hashes'][case['id']] == case['source_sha256'], 'Source replay mismatch')
            require(sha(prefix/case['panel']) == case['panel_sha256'], 'Replay panel has changed')
            for variant in case['candidates']:
                require(sha(ROOT/variant['path']) == variant['sha256'], 'Replay image has changed')
        runs[name] = {c['id']: c for c in replay['cases']}
    notes = {c['id']: c for c in review['notes']}
    records = []
    for spec in plan['cases']:
        cid = spec['id']
        row = sources[cid]
        note = notes[cid]
        for panel, digest in note['reviewed_panels'].items():
            require(sha(EXP/panel) == digest, 'Visually reviewed pixels have changed')
        source = ROOT/row['source']
        require(sha(source) == iso['candidate_source_hashes'][cid], 'Original photograph has changed')
        inputs = DATA/'inputs'
        inputs.mkdir(parents=True, exist_ok=True)
        image = inputs/(cid+source.suffix)
        shutil.copyfile(source, image)
        require(sha(image) == iso['candidate_source_hashes'][cid], 'Input copy failed')
        case = dict(id=cid, group_id=row['group_id'], split='train_candidate', domain=spec['domain'],
                    image=str(image.relative_to(ROOT)), image_sha256=sha(image), source=row['source'],
                    source_annotation_indices=row['annotation_indices'], source_partition='training_pool',
                    instructions=spec['instructions'], instruction_origin='AI-authored paired requests; GIER TRAIN photographs',
                    observation=note['observation'], panels=note['reviewed_panels'])
        choice = note['decision_candidate']
        rgb = load_rgb(image)
        replay = runs['initial'][cid]['candidates']
        if isinstance(choice, int):
            require(note['initial_edit_verdicts'][choice-1] == 'complete', 'Chosen replay must be complete')
            target = replay[choice]
            pixels = execute(rgb, target['decision'])['result']
            require(np.array_equal(pixels, load_rgb(ROOT/target['path'])), 'Target pixels do not reproduce')
            case.update(status='visual_edit_candidate', decision=target['decision'],
                        target=target['path'], target_sha256=target['sha256'])
            rejected = note['rejected_for_preference']
            if rejected is not None:
                require(note['initial_edit_verdicts'][rejected-1] == 'failed', 'Rejected replay must be a visible failure')
                variant = replay[rejected]
                case['preference_candidate'] = dict(chosen=target['decision'], rejected=variant['decision'],
                                                    rejected_result=variant['path'], rejected_result_sha256=variant['sha256'])
        elif choice == 'decline_candidate':
            require(all(verdict == 'failed' for verdict in note['initial_edit_verdicts'] + note['refined_edit_verdicts']),
                    'A decline candidate needs visible failure across its tested alternatives')
            decision = dict(action='decline', reason='unreliable_quality')
            require(execute(rgb, decision)['result'] is None, 'Decline unexpectedly returned an image')
            case.update(status='provisional_decline_candidate', decision=decision, target=None)
        else:
            case.update(status='uncertain_no_supervision', decision=None, target=None)
        records.append(case)
    require(len({c['group_id'] for c in records}) == len(records), 'Candidate groups are not distinct')
    paired = [
        dict(case_id=c['id'], group_id=c['group_id'], language=lang, image=c['image'], status=c['status'],
             instruction=text, decision=c['decision'], assistant_json=target_text(c['decision']))
        for c in records if c['decision'] is not None
        for lang, text in c['instructions'].items()
    ]
    write(DATA/'reviewed-candidates.json', records)
    write(DATA/'paired-candidate-labels.json', paired)
    write(DATA/'receipt.json', dict(at=stamp(), status='staged_for_review_not_admitted_to_training',
        source_review_sha256=sha(EXP/'visual-review.json'), isolation_sha256=sha(EXP/'isolation.json'),
        candidates_sha256=sha(DATA/'reviewed-candidates.json'), paired_labels_sha256=sha(DATA/'paired-candidate-labels.json'),
        input_groups=len(records), image_edit_candidates=sum(c['status']=='visual_edit_candidate' for c in records),
        provisional_decline_candidates=sum(c['status']=='provisional_decline_candidate' for c in records),
        uncertain_cases=sum(c['decision'] is None for c in records),
        preference_cases=sum('preference_candidate' in c for c in records),
        missing_official_comparison_images=len(iso['missing']), GPU_hours=0, trained_models=0,
        independent_Qwen_review=False, independent_user_review=False, acceptance_opened=False))
    print(f'Staged {len(records)} source groups, {len(paired)} bilingual candidate labels')


if __name__ == '__main__':
    main()

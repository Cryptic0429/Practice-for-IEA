"""Report frozen reward-model calibration; never equate this to useful editing."""
import json
from collections import Counter

from functional_v1.common import ROOT, read, write, sha, stamp, GPUJob
from .prepare_calibration import EXP, DATA


def main():
    protocol = read(EXP / 'calibration-protocol.json')
    clarification = read(EXP / 'score-dimension-clarification.json')
    assert clarification['raw_scores_or_predictions_inspected'] is False
    assert clarification['protocol_sha256'] == sha(EXP / 'calibration-protocol.json')
    dataset = read(DATA / 'calibration-candidates.json')
    summaries = {}
    for route in ('base', 'editscore'):
        folder = EXP / 'reward-calibration' / route
        state = read(folder / 'status.json')
        assert state['status'] == 'completed' and state['completed'] == 76
        rows = list(map(json.loads, (folder / 'predictions.jsonl').read_text().splitlines()))
        assert len(rows) == len({r['id'] for r in rows}) == 76
        lookup = {(r['case_id'], r['language'], r['role']): r for r in rows}
        direction = []
        for case in dataset['direction']:
            languages = []
            for language in ('zh', 'en'):
                found = {role: lookup[case['id'], language, role] for role in ('admitted', 'identity', 'opposite')}
                valid = all(r['status'] == 'scored' for r in found.values())
                scores = {role: r['score'][0] if r['status'] == 'scored' else None for role, r in found.items()}
                comparisons = {bad: bool(valid and scores['admitted'] > scores[bad]) for bad in ('identity', 'opposite')}
                languages.append(dict(language=language, edit_success_scores=scores,
                                      passed=all(comparisons.values()), comparisons=comparisons,
                                      consistency_scores={role: r['score'][1] if r['status'] == 'scored' else None for role, r in found.items()}))
            direction.append(dict(id=case['id'], family=case['family'], domain=case['domain'],
                                  passed_both_languages=all(r['passed'] for r in languages), languages=languages))
        quality = []
        for case in dataset['quality']:
            found = {role: lookup[case['id'], None, role] for role in ('admitted', 'damaged')}
            valid = all(r['status'] == 'scored' for r in found.values())
            scores = {role: min(r['score']) if r['status'] == 'scored' else None for role, r in found.items()}
            quality.append(dict(id=case['id'], domain=case['domain'], min_naturalness_artifacts=scores,
                                passed=bool(valid and scores['admitted'] > scores['damaged']),
                                severity=case['bad_severity'], scope=case['scope']))
        d = sum(r['passed_both_languages'] for r in direction)
        q = sum(r['passed'] for r in quality)
        gate = protocol['gates']
        summary = dict(at=stamp(), route=route, status='completed',
                       direction_tasks_both_languages_passed=d, direction_tasks=10,
                       quality_pairs_passed=q, quality_pairs=8,
                       invalid_outputs=sum(r['status'] != 'scored' for r in rows),
                       passed_calibration=d >= gate['direction_semantic_tasks_required'] and q >= gate['quality_pairs_required'],
                       direction_results=direction, quality_results=quality,
                       photo_groups=18, photo_domains={'portrait': 9, 'landscape': 9},
                       assessment='Agreement with fixed small direct-AI training-pool calibration, not independent human evaluation or practical editor usefulness.',
                       own_policy_training=False, editor_changed=False, acceptance_opened=False,
                       protocol_sha256=sha(EXP / 'calibration-protocol.json'),
                       dimension_clarification_sha256=sha(EXP / 'score-dimension-clarification.json'),
                       predictions_sha256=sha(folder / 'predictions.jsonl'),
                       caveat='Score dimension clarification was saved after some outputs were written but before any raw score inspection; exact counts and timestamp are disclosed in its receipt. No thresholds, data, prompt or generation changed.')
        write(folder / 'calibration-summary.json', summary)
        summaries[route] = summary
    result = dict(at=stamp(), summaries=summaries,
                  selected_training_reward='editscore' if summaries['editscore']['passed_calibration'] else None,
                  next_gate='Additional candidate verification still requires direct images; calibration alone never promotes a policy.',
                  stop_this_configuration_as_training_reward=not summaries['editscore']['passed_calibration'],
                  policy_training_started=False,
                  resource_snapshot=dict(occupied_gpu_hours=GPUJob.consumed(read(ROOT / 'experiments/functional-v1-20261104/resources.json')), limit=16))
    write(EXP / 'calibration-decision.json', result)
    print(json.dumps({route: {k: s[k] for k in ('direction_tasks_both_languages_passed', 'quality_pairs_passed', 'invalid_outputs', 'passed_calibration')} for route, s in summaries.items()}))


if __name__ == '__main__':
    main()

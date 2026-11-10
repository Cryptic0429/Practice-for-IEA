"""Explain why a score threshold alone cannot fix the failed artifact screen."""
from collections import Counter

from functional_v1.common import read, sha, stamp, write
from .run import EXP


def main():
    fit_path = EXP/'fit.json'
    eval_path = EXP/'evaluation.json'
    fitted, evaluation = read(fit_path), read(eval_path)
    if evaluation['fitted_sha256'] != sha(fit_path):
        raise ValueError('Evaluation refers to another fitted screen')
    rows = evaluation['examples']
    good = [r for r in rows if r['artifact'] is False]
    bad = [r for r in rows if r['artifact'] is True]
    scene = {r['id']: r for r in rows if r['cohort'] == 'r24_stored_original'}
    diagnostics = {}
    for name, cohorts in evaluation['summary'].items():
        def risk(row):
            return row['screens'][name]['probability']

        worst_good, highest_bad, lowest_bad = max(good, key=risk), max(bad, key=risk), min(bad, key=risk)
        false_blocks = [r for r in good if risk(r) >= risk(lowest_bad)]
        missed = sum(c['artifact_allowed']['rows'] for c in cohorts.values())
        blocked = sum(c['usable_blocked']['rows'] for c in cohorts.values())
        diagnostics[name] = dict(
            allowed_known_artifact_rows=missed, blocked_known_usable_rows=blocked,
            passes_prespecified_screen=missed == 0 and blocked == 0,
            highest_usable=dict(id=worst_good['id'], score=risk(worst_good)),
            highest_artifact=dict(id=highest_bad['id'], score=risk(highest_bad)),
            artifacts_above_every_usable=sum(risk(r)>risk(worst_good) for r in bad),
            lowest_artifact=dict(id=lowest_bad['id'], score=risk(lowest_bad)),
            usable_rows_blocked_if_all_artifacts_blocked=len(false_blocks),
            usable_distinct_renders_blocked_if_all_artifacts_blocked=len({
                (r['input_sha256'],r['output_sha256']) for r in false_blocks}),
            p227_usable_score=risk(scene['p227-initial-1']),
            p227_artifact_score=risk(scene['p227-initial-3']))
    witness_mix = {}
    for name, tag in [('usable', False), ('artifact', True)]:
        count = Counter(tuple(sorted(r['parameters'].items())) for r in fitted['examples'] if r['artifact'] is tag)
        witness_mix[name] = [dict(parameters=dict(p), count=n) for p,n in sorted(count.items())]
    write(EXP/'diagnostic.json', dict(
        at=stamp(), evaluation_sha256=sha(eval_path), fitted_sha256=sha(fit_path),
        diagnostic_only_not_threshold_tuning=True, witness_parameter_mix=witness_mix,
        screens=diagnostics, deployed=False, GPU_hours=0, final_acceptance_opened=False))


if __name__ == '__main__':
    main()

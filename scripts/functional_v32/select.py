"""Fit a small CPU contrast candidate selector on grouped visual judgements."""
import argparse
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from functional_v1.common import ROOT, read, sha, stamp, write
from functional_v15.editor import load_rgb
from functional_v32.editor import LUMA, RECIPES
from functional_v32.render import DATA, EXP

FEATURES = (
    'original_median', 'original_luma_span', 'original_dark_share', 'original_bright_share',
    'signed_span_change', 'signed_middle_span_change', 'absolute_median_change',
    'shadow_spread_loss', 'highlight_spread_loss', 'new_black', 'new_white', 'mean_chroma_change',
)


def describe(original, edited, direction):
    before = original.astype(np.float32) / 255
    after = edited.astype(np.float32) / 255
    y, z = before @ LUMA, after @ LUMA
    b = np.quantile(y, [.02, .10, .25, .50, .75, .90, .98])
    a = np.quantile(z, [.02, .10, .25, .50, .75, .90, .98])
    sign = 1 if direction == 'increase' else -1
    span_change = (a[5] - a[1]) - (b[5] - b[1])
    middle_change = (a[4] - a[2]) - (b[4] - b[2])
    shadow_loss = (b[2] - b[0]) - (a[2] - a[0])
    highlight_loss = (b[6] - b[4]) - (a[6] - a[4])
    return [float(v) for v in (
        b[3], b[5] - b[1], np.mean(y < .10), np.mean(y > .90),
        sign * span_change, sign * middle_change, abs(a[3] - b[3]),
        shadow_loss, highlight_loss,
        np.mean((original.max(axis=2) > 1) & (edited.max(axis=2) <= 1)),
        np.mean((original.min(axis=2) < 254) & (edited.min(axis=2) >= 254)),
        np.mean(np.abs(np.ptp(after, axis=2) - np.ptp(before, axis=2))),
    )]


def build(cohort):
    review_path = EXP / f'visual-{cohort}.json'
    judgements = read(review_path)
    replay = EXP / 'replay.json'
    rows = []
    for case in read(replay)['cases']:
        if case['cohort'] != cohort:
            continue
        if len(judgements[case['id']]) != len(RECIPES):
            raise ValueError(f"Incomplete visual observations for {case['id']}")
        source_path = ROOT / case['source']
        if sha(source_path) != case['source_sha256']:
            raise ValueError(f"Original changed for {case['id']}")
        original = load_rgb(source_path)
        for recipe, obs in zip(case['candidate_order'], judgements[case['id']], strict=True):
            image = case['images'][recipe]
            output_path = ROOT / image['path']
            if sha(output_path) != image['sha256']:
                raise ValueError(f"Output changed for {case['id']} {recipe}")
            rows.append(dict(case_id=case['id'], group_id=case['group_id'], photo_id=case['photo_id'],
                             domain=case['domain'], direction=case['direction'], requests=case['requests'],
                             source=case['source'], source_sha256=case['source_sha256'],
                             output=image, recipe=recipe, outcome=obs[0], reason=obs[1],
                             features=describe(original, load_rgb(output_path), case['direction'])))
    if set(judgements) != {r['case_id'] for r in rows}:
        raise ValueError('Visual observations do not match replay')
    path = DATA / f'{cohort}-candidate-records.json'
    write(path, dict(replay_sha256=sha(replay), review_sha256=sha(review_path), features=FEATURES, rows=rows))
    return path


def new_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=.25, class_weight='balanced', max_iter=500))


def arrays(rows):
    return np.asarray([r['features'] for r in rows]), np.asarray([r['outcome'] == 'complete' for r in rows])


def grouped(rows, scores):
    by_case = defaultdict(list)
    for row, score in zip(rows, scores, strict=True):
        by_case[row['case_id']].append(dict(row, score=float(score)))
    return by_case


def fit():
    path = build('fit')
    rows = read(path)['rows']
    x, y = arrays(rows)
    oof = np.zeros(len(rows))
    for train, held in LeaveOneGroupOut().split(x, y, [r['group_id'] for r in rows]):
        model = new_model().fit(x[train], y[train])
        oof[held] = model.decision_function(x[held])
    winners = [max(cases, key=lambda r: r['score']) for cases in grouped(rows, oof).values()]
    bad = [r['score'] for r in winners if r['outcome'] != 'complete']
    # Accept only scores above every out-of-group unsafe winner.
    threshold = max(bad) if bad else min(r['score'] for r in winners) - 1e-6
    model = new_model().fit(x, y)
    scaler, head = model.steps[0][1], model.steps[1][1]
    write(EXP / 'fit.json', dict(at=stamp(), training_path=str(path.relative_to(ROOT)),
          training_sha256=sha(path), source_groups=len({r['group_id'] for r in rows}),
          candidate_count=len(rows), threshold=threshold, features=FEATURES,
          estimator='StandardScaler, balanced logistic regression, C=.25, 8 leave-one-original-group-out folds',
          labels=dict(Counter(r['outcome'] for r in rows)),
          mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
          weights=head.coef_[0].tolist(), bias=float(head.intercept_[0]),
          oof_winners=[{k: r[k] for k in ('case_id', 'recipe', 'score', 'outcome')} for r in winners],
          GPU_hours=0))


def evaluate():
    fitted = read(EXP / 'fit.json')
    if sha(ROOT / fitted['training_path']) != fitted['training_sha256']:
        raise ValueError('Fit supervision changed')
    path = build('held')
    rows = read(path)['rows']
    scores = ((np.asarray([r['features'] for r in rows]) - fitted['mean']) / fitted['scale']) @ fitted['weights'] + fitted['bias']
    results, totals = {}, defaultdict(Counter)
    for case, candidates in grouped(rows, scores).items():
        oracle = any(r['outcome'] == 'complete' for r in candidates)
        winner = max(candidates, key=lambda r: r['score'])
        def outcome(r):
            return {k: r[k] for k in ('recipe', 'outcome', 'reason', 'score')}
        methods = {r['recipe']: outcome(r) for r in candidates}
        methods['selector'] = outcome(winner) if winner['score'] > fitted['threshold'] else {
            'recipe': None, 'outcome': 'missed' if oracle else 'defer',
            'top_recipe': winner['recipe'], 'score': winner['score']}
        for name, v in methods.items():
            totals[name][v['outcome']] += 1
        results[case] = dict(oracle_complete=oracle, domain=winner['domain'], methods=methods)
    write(EXP / 'result.json', dict(at=stamp(), fit_sha256=sha(EXP / 'fit.json'),
          held_sha256=sha(path), held_cases=len(results), held_originals=len({r['group_id'] for r in rows}),
          oracle_complete=sum(r['oracle_complete'] for r in results.values()), totals=dict(totals),
          both_directions_complete={name: sum(all(results[f"{p}-{direction}"]['methods'][name]['outcome'] == 'complete'
              for direction in ('increase', 'decrease')) for p in {r['photo_id'] for r in rows})
              for name in totals}, results=results, GPU_hours=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=('fit', 'evaluate'))
    args = parser.parse_args()
    (fit if args.step == 'fit' else evaluate)()

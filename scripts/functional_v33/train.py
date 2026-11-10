"""Train directional contrast selectors and evaluate only after freezing their weights."""
import argparse
from collections import Counter, defaultdict
from itertools import product

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from functional_v1.common import ROOT, read, sha, stamp, write
from functional_v15.editor import load_rgb
from functional_v31.face import FaceRegions
from functional_v32.editor import RECIPES
from functional_v32.select import FEATURES, describe
from functional_v33.cohort import DATA, EXP, PLAN, R32, R32_DATA
from functional_v33.spatial import SPATIAL, SourceFeatures

DIRECTIONS = ('increase', 'decrease')
OUTCOMES = ('complete', 'partial', 'failed', 'uncertain')
METHODS = {'pool_refit': (False, False, False), 'split_global': (True, False, False),
           'split_spatial': (True, True, False), 'split_pair': (True, True, True)}


def checked(entry):
    path = ROOT / entry['path']
    if sha(path) != entry['sha256']:
        raise ValueError(f'Image bytes changed: {path}')
    return path


def cases(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['case_id']].append(row)
    if any({r['recipe'] for r in rows} != set(RECIPES) or len(rows) != len(RECIPES) for rows in groups.values()):
        raise ValueError('Each request must have exactly the frozen recipe grid')
    return groups


def make_rows(cohort):
    detector = FaceRegions()
    if cohort == 'fit':
        sources = [R32_DATA / f'{c}-candidate-records.json' for c in ('fit', 'held')]
        r32fit, r32result = read(R32 / 'fit.json'), read(R32 / 'result.json')
        if [sha(p) for p in sources] != [r32fit['training_sha256'], r32result['held_sha256']]:
            raise ValueError('R32 visual supervision changed')
        incoming = [r for path in sources for r in read(path)['rows']]
    else:
        sources = [EXP / 'replay.json', EXP / 'visual-held.json']
        replay, judgements = map(read, sources)
        if replay['protocol_sha256'] != sha(PLAN):
            raise ValueError('Holdout protocol changed')
        if set(judgements) != {c['id'] for c in replay['cases']}:
            raise ValueError('Missing visual judgement')
        incoming = []
        for c in replay['cases']:
            for recipe, obs in zip(c['candidate_order'], judgements[c['id']], strict=True):
                incoming.append(dict(case_id=c['id'], group_id=c['group_id'], photo_id=c['photo_id'],
                                     domain=c['domain'], light=c['light'], direction=c['direction'],
                                     requests=c['requests'], source=c['source'], source_sha256=c['source_sha256'],
                                     output=c['images'][recipe], recipe=recipe, outcome=obs[0], reason=obs[1]))
    cases(incoming)
    source_cache = None
    for row in incoming:
        if row['outcome'] not in OUTCOMES:
            raise ValueError(f"Invalid outcome for {row['case_id']}")
        if row['source'] != source_cache:
            source_cache = row['source']
            original = load_rgb(checked({'path': source_cache, 'sha256': row['source_sha256']}))
            source = SourceFeatures(original, detector)
        edited = load_rgb(checked(row['output']))
        if original.shape != edited.shape:
            raise ValueError('Candidate does not align with source')
        if cohort == 'held':
            row['features'] = describe(original, edited, row['direction'])
        row['spatial'] = source.compare(edited)
        row['face_boxes'] = source.boxes
    path = DATA / f'{cohort}-candidate-records.json'
    write(path, dict(sources={str(p.relative_to(ROOT)): sha(p) for p in sources},
                     global_feature_names=FEATURES, spatial_feature_names=SPATIAL, rows=incoming))
    return path


def vector(row, spatial):
    return row['features'] + (row['spatial'] if spatial else [])


def trained(rows, spatial, paired):
    x = np.asarray([vector(r, spatial) for r in rows])
    scaler = StandardScaler().fit(x)
    fit_x = scaler.transform(x)
    y = np.asarray([r['outcome'] == 'complete' for r in rows])
    comparisons = 0
    if paired:
        by_case = defaultdict(list)
        for row, value, ok in zip(rows, fit_x, y, strict=True):
            by_case[row['case_id']].append((value, ok))
        differences = []
        for group in by_case.values():
            differences.extend(a - b for a, b in product(
                (v for v, ok in group if ok), (v for v, ok in group if not ok)))
        diff = np.asarray(differences)
        comparisons = len(diff)
        fit_x = np.concatenate((diff, -diff))
        y = np.repeat((True, False), comparisons)
    model = LogisticRegression(C=.25, class_weight='balanced', max_iter=500).fit(fit_x, y)
    return dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), weights=model.coef_[0].tolist(),
                bias=float(model.intercept_[0]), paired_comparisons=comparisons)


def scores(rows, head):
    spatial = len(head['weights']) > len(FEATURES)
    x = np.asarray([vector(row, spatial) for row in rows])
    return ((x - head['mean']) / head['scale']) @ head['weights'] + head['bias']


def winner(rows, rank, quality):
    choices = scores(rows, rank)
    idx = max(range(len(rows)), key=lambda i: (choices[i], -RECIPES.index(rows[i]['recipe'])))
    return rows[idx], float(scores([rows[idx]], quality)[0]), float(choices[idx])


def heads(rows, spatial, paired):
    pointwise = trained(rows, spatial, False)
    return trained(rows, spatial, True) if paired else pointwise, pointwise


def fit_method(rows, split, spatial, paired):
    fitted = {}
    for direction in DIRECTIONS if split else ('all',):
        selected = [r for r in rows if direction in ('all', r['direction'])]
        groups = {r['group_id'] for r in selected}
        folds = []
        for group in sorted(groups):
            train = [r for r in selected if r['group_id'] != group]
            val = [r for r in selected if r['group_id'] == group]
            rank, quality = heads(train, spatial, paired)
            for key, candidates in cases(val).items():
                row, score, ranking = winner(candidates, rank, quality)
                folds.append(dict(case_id=key, group_id=group, recipe=row['recipe'],
                                  outcome=row['outcome'], score=score, rank=ranking))
        bad = [r['score'] for r in folds if r['outcome'] != 'complete']
        threshold = max(bad) if bad else min(r['score'] for r in folds) - 1e-6
        rank, quality = heads(selected, spatial, paired)
        fitted[direction] = dict(ranker=rank, quality=quality, threshold=threshold, oof_winners=folds,
                                 oof_accepted=sum(r['score'] > threshold for r in folds))
    return fitted


def fit():
    path = make_rows('fit')
    rows = read(path)['rows']
    groups = {r['group_id'] for r in rows}
    if len(groups) != 16 or len(rows) != 128:
        raise ValueError('Training cohort differs from the frozen data plan')
    models = {name: fit_method(rows, *config) for name, config in METHODS.items()}
    write(EXP / 'fit.json', dict(at=stamp(), training_path=str(path.relative_to(ROOT)), training_sha256=sha(path),
          protocol_sha256=sha(PLAN), source_groups=len(groups), candidates=len(rows),
          models=models, frozen_control_sha256=sha(R32 / 'fit.json'), GPU_hours=0))


def outcome(row, score=None, rank=None):
    return dict(recipe=row['recipe'], outcome=row['outcome'], reason=row['reason'], score=score, rank=rank)


def decide(candidates, rank, quality, threshold):
    row, score, ranking = winner(candidates, rank, quality)
    decision = outcome(row, score, ranking)
    if score <= threshold:
        decision.update(recipe=None, top_recipe=row['recipe'], top_outcome=row['outcome'],
                        outcome='missed' if any(r['outcome'] == 'complete' for r in candidates) else 'defer')
    return decision


def evaluate():
    frozen = read(EXP / 'fit.json')
    if frozen['protocol_sha256'] != sha(PLAN) or frozen['training_sha256'] != sha(ROOT / frozen['training_path']):
        raise ValueError('Frozen training inputs changed')
    if frozen['frozen_control_sha256'] != sha(R32 / 'fit.json'):
        raise ValueError('Frozen R32 model changed')
    path = make_rows('held')
    rows = read(path)['rows']
    fit_groups = {r['group_id'] for r in read(ROOT / frozen['training_path'])['rows']}
    held_groups = {r['group_id'] for r in rows}
    if fit_groups & held_groups or len(held_groups) != 12 or len(rows) != 96:
        raise ValueError('Holdout groups differ from the frozen plan')
    old = read(R32 / 'fit.json')
    results, totals, directions, raw = {}, defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for key, group in cases(rows).items():
        direction = group[0]['direction']
        by_recipe = {r['recipe']: r for r in group}
        methods = {'fixed20': outcome(by_recipe['r15_20']),
                   'fixed_split': outcome(by_recipe['protected_strong' if direction == 'increase' else 'r15_20']),
                   'r32_frozen': decide(group, old, old, old['threshold'])}
        for name, settings in frozen['models'].items():
            head = settings.get(direction, settings.get('all'))
            methods[name] = decide(group, head['ranker'], head['quality'], head['threshold'])
        for name, value in methods.items():
            totals[name][value['outcome']] += 1
            directions[name][f"{direction}:{value['outcome']}"] += 1
            raw[name][value.get('top_outcome', value['outcome'])] += 1
        results[key] = dict(domain=group[0]['domain'], light=group[0]['light'], direction=direction,
                            oracle_complete=any(r['outcome'] == 'complete' for r in group),
                            methods=methods, candidates=[outcome(r) for r in group])
    write(EXP / 'result.json', dict(at=stamp(), fit_sha256=sha(EXP / 'fit.json'), held_sha256=sha(path),
          held_cases=len(results), held_originals=len(held_groups), totals=dict(totals), raw_tops=dict(raw),
          by_direction=dict(directions), oracle_complete=sum(r['oracle_complete'] for r in results.values()),
          both_directions_complete={name: sum(all(results[f"{p}-{direction}"]['methods'][name]['outcome'] == 'complete'
              for direction in DIRECTIONS) for p in {r['photo_id'] for r in rows}) for name in totals},
          results=results, GPU_hours=0))


if __name__ == '__main__':
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=('fit', 'evaluate'))
    args = parser.parse_args()
    (fit if args.step == 'fit' else evaluate)()

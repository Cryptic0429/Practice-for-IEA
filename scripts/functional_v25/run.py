"""Fit two CPU artifact screens and test on source-disjoint saved image pairs."""
import argparse
from collections import Counter
import json
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from functional_v1.common import ROOT, read, sha, stamp, write
from functional_v15.outcomes import EDITOR_VERSION
from .features import INPUT_NAMES, PAIR_NAMES, features, positive_tonal_edit

EXP = ROOT/'experiments/functional-v25-20261107'
R22 = ROOT/'experiments/functional-v22-20261107'
R24 = ROOT/'experiments/functional-v24-20261107'
R24_DATA = ROOT/'data/training/functional-v24-20261107'
SOURCE_FILES = {}
SEED = 20261107


def source(path, lines=False):
    SOURCE_FILES[str(path.relative_to(ROOT))] = sha(path)
    if lines:
        return [json.loads(line) for line in path.read_text().splitlines()]
    return read(path)


def item(set_name, cid, group, inp, inp_sha, out, out_sha, decision, expected, evidence, **meta):
    if not positive_tonal_edit(decision):
        raise ValueError(f'Unsupported edit in artifact cohort: {cid}')
    return dict(cohort=set_name, id=cid, group_id=group, input=inp, input_sha256=inp_sha,
                output=out, output_sha256=out_sha, parameters=decision['parameters'],
                artifact=expected, evidence=evidence, **meta)


def synthetic():
    manifest = R22/'quality-candidates/manifest.json'
    notes_path = R22/'quality-native-notes.jsonl'
    admission = source(ROOT/'data/training/functional-v22-20261107/admission.json')
    notes = {row['id']: row for row in source(notes_path, lines=True)}
    cases = source(manifest)['items']
    for path in (manifest, notes_path):
        key = str(path.relative_to(ROOT))
        if admission['evidence'][key] != SOURCE_FILES[key]:
            raise ValueError(f'R22 evidence changed: {key}')
    for case in cases:
        note = notes[case['id']]
        if note['disposition'] == 'exclude_uncertain':
            continue
        artifact = note['disposition'] == 'accept_decline'
        if artifact and not re.search(r'noise|grain|speckl|stippl|mottl|blotch', note['evidence'], re.I):
            continue
        if not artifact and note['disposition'] != 'accept_edit':
            continue
        out = next(a for a in case['candidates'] if a['id'] == note['candidate_id'])
        if case['input_sha256'] != note['input_sha256'] or out['sha256'] != note['candidate_sha256']:
            raise ValueError(f'R22 native witness mismatch: {case["id"]}')
        yield item('r22_synthetic', case['id'], case['group_id'], case['image'], case['input_sha256'],
                   out['path'], out['sha256'], out['decision'], artifact, note['evidence'])


def stored_originals():
    receipt = source(R24_DATA/'receipt.json')
    cases_path, review_path = R24_DATA/'reviewed-candidates.json', R24/'visual-review.json'
    cases = {c['id']: c for c in source(cases_path)}
    review = source(review_path)
    if (SOURCE_FILES[str(cases_path.relative_to(ROOT))] != receipt['candidates_sha256']
        or SOURCE_FILES[str(review_path.relative_to(ROOT))] != receipt['source_review_sha256']):
        raise ValueError('R24 evidence changed')
    notes = {n['id']: n for n in review['notes']}
    negatives = {'initial': {'p227': [3], 'p444': [3], 'p425': [3], 'p389': [3]},
                 'refined': {'p425': [1, 2], 'p389': [1, 2]}}
    for stage, path in [('initial', R24/'replay.json'), ('refined', R24/'refinements/replay.json')]:
        replays = source(path)
        if SOURCE_FILES[str(path.relative_to(ROOT))] != review[f'{stage}_replay_sha256']:
            raise ValueError('R24 replay manifest changed')
        for replay in replays['cases']:
            cid = replay['id']
            case, note = cases[cid], notes[cid]
            if case['image_sha256'] != replay['source_sha256']:
                raise ValueError(f'R24 replay input mismatch: {cid}')
            for output in replay['candidates'][1:]:
                i = output['index']
                artifact = i in negatives[stage].get(cid, [])
                complete = stage == 'initial' and i == note['decision_candidate']
                if not artifact and not complete:
                    continue
                if artifact and note[f'{stage}_edit_verdicts'][i-1] != 'failed':
                    raise ValueError(f'Unconfirmed R24 artifact witness: {cid} {stage} {i}')
                yield item('r24_stored_original', f'{cid}-{stage}-{i}', case['group_id'],
                           case['image'], case['image_sha256'], output['path'], output['sha256'],
                           output['decision'], artifact, note['observation'])


def history():
    directory = ROOT/'experiments/functional-v23-20261107/evaluation'/EDITOR_VERSION
    for split in ('native-development', 'old-native'):
        for seed in (20261207, 20261208):
            yield from historical_pair(directory/split/f'lora-seed-{seed}-step250', 'r23_historical', split)
    path = ROOT/'experiments/functional-v21-20261107/replication/evaluation'/EDITOR_VERSION
    yield from historical_pair(path/'native-development/magnitude-quality-e20', 'r21_stress', 'native-development')


def historical_pair(path, cohort, split):
    evidence = {(r['case_id'], r['language']): r['assessment'] for r in source(path/'direct-review.json')}
    for row in source(path/'predictions.jsonl', lines=True):
        key = row['case_id'], row['language']
        if cohort == 'r21_stress' and key != ('native-dev-070', 'zh'):
            continue
        if row.get('execution_status') != 'edited' or not positive_tonal_edit(row['decision']):
            continue
        note = evidence[key]
        if row['output_sha256'] != note['image_sha256']:
            raise ValueError(f'Historical output hash mismatch: {key}')
        if cohort == 'r21_stress':
            if note['quality'] != 'major_issue':
                raise ValueError('R21 fixed major artifact probe mismatch')
            expected = True
        else:
            expected = False if (note['completion'] == 'completed'
                                 and note['quality'] in ('acceptable', 'minor_issue')) else None
        yield item(cohort, f"{path.name}/{key[0]}-{key[1]}", row['group_id'],
                   row['input_image'], row['input_sha256'], row['output_image'], row['output_sha256'],
                   row['decision'], expected, note['evidence'], language=key[1], case_id=key[0], split=split)


def matrix(items):
    cache, rows = {}, []
    for row in items:
        key = row['input_sha256'], row['output_sha256']
        inp, out = ROOT/row['input'], ROOT/row['output']
        if sha(inp) != key[0] or sha(out) != key[1]:
            raise ValueError(f'Image file integrity check failed: {row["id"]}')
        if key not in cache:
            cache[key] = features(inp, out)
        rows.append(cache[key].tolist())
    return np.asarray(rows)


def estimator():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=.3, class_weight='balanced', solver='liblinear', max_iter=1000, random_state=SEED))


def counts(labels, flags, items):
    result = {}
    for name, mask in [('usable_blocked', (labels == 0) & flags),
                       ('usable_allowed', (labels == 0) & ~flags),
                       ('artifact_blocked', (labels == 1) & flags),
                       ('artifact_allowed', (labels == 1) & ~flags),
                       ('unscored_blocked', (labels == -1) & flags),
                       ('unscored_allowed', (labels == -1) & ~flags)]:
        selected = [r for r, include in zip(items, mask) if include]
        result[name] = dict(rows=len(selected), unique_groups=len({r['group_id'] for r in selected}),
                            distinct_render_pairs=len({(r['input_sha256'], r['output_sha256']) for r in selected}))
    return result


def fit(items):
    x = matrix(items)
    y = np.array([int(r['artifact']) for r in items])
    groups = [r['group_id'] for r in items]
    splits = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED).split(x,y,groups))
    models = {}
    for name, cols in [('input_only', INPUT_NAMES), ('paired', PAIR_NAMES)]:
        k = len(cols)
        oof = np.empty(len(y))
        for train, test in splits:
            if set(groups[i] for i in train) & set(groups[i] for i in test):
                raise ValueError('Source-group leakage between fit folds')
            oof[test] = estimator().fit(x[train, :k], y[train]).predict_proba(x[test, :k])[:, 1]
        candidates = np.append(np.unique(oof), np.nextafter(oof.max(), np.inf))
        choices = []
        for threshold in candidates:
            blocked = oof >= threshold
            fp = int(sum(blocked & (y == 0)))
            tp = int(sum(blocked & (y == 1)))
            if fp <= 1:
                choices.append((tp, -fp, float(threshold)))
        _, _, threshold = max(choices)
        model = estimator().fit(x[:, :k], y)
        scale, weights = model.steps[0][1], model.steps[1][1]
        models[name] = dict(feature_names=cols, threshold=threshold,
                            mean=scale.mean_.tolist(), scale=scale.scale_.tolist(),
                            coefficients=weights.coef_[0].tolist(), intercept=float(weights.intercept_[0]),
                            calibration_counts=counts(y, oof >= threshold, items),
                            oof_rows=[dict(id=r['id'], probability=float(p)) for r, p in zip(items, oof)])
    return dict(at=stamp(), protocol_sha256=sha(EXP/'protocol.json'), data_sources=SOURCE_FILES,
                examples=items, feature_rows=x.tolist(), models=models, GPU_hours=0)


def evaluate(trained, items):
    x = matrix(items)
    y = np.array([-1 if r['artifact'] is None else int(r['artifact']) for r in items])
    fits = trained['models']
    summary, score_rows = {}, [dict(r, screens={}) for r in items]
    for name, model in fits.items():
        z = (x[:, :len(model['feature_names'])] - model['mean']) / model['scale']
        logits = z @ model['coefficients'] + model['intercept']
        scores = 1/(1+np.exp(-np.clip(logits,-100,100)))
        flags = scores >= model['threshold']
        result = {}
        for key in sorted({r['cohort'] for r in items}):
            mask = np.array([r['cohort'] == key for r in items])
            result[key] = counts(y[mask], flags[mask], [r for r in items if r['cohort'] == key])
        summary[name] = result
        for row, score, blocked in zip(score_rows, scores, flags):
            row['screens'][name] = dict(probability=float(score), blocked=bool(blocked))
    return dict(at=stamp(), fitted_sha256=sha(EXP/'fit.json'), data_sources=SOURCE_FILES,
                summary=summary, examples=score_rows, feature_rows=x.tolist(), GPU_hours=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('fit','evaluate'))
    args = parser.parse_args()
    if args.stage == 'fit':
        if (EXP/'fit.json').exists():
            raise FileExistsError('Keep the first fitted screen frozen')
        items = list(synthetic())
        print('Synthetic conditions:', dict(Counter(r['artifact'] for r in items)), flush=True)
        write(EXP/'fit.json', fit(items))
    else:
        trained = read(EXP/'fit.json')
        if trained['protocol_sha256'] != sha(EXP/'protocol.json'):
            raise ValueError('Protocol has changed since fitting')
        items = [*stored_originals(), *history()]
        if {r['group_id'] for r in items} & {r['group_id'] for r in trained['examples']}:
            raise ValueError('Evaluation shares an original group with training')
        print('Test groups:', dict(Counter((r['cohort'], r['artifact']) for r in items)), flush=True)
        write(EXP/'evaluation.json', evaluate(trained, items))


if __name__ == '__main__':
    main()

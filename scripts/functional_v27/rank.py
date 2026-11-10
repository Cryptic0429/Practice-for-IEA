"""Run a locked conditional ranking comparison without GPU inference."""
import argparse
from collections import Counter
from itertools import combinations
import shutil

import numpy as np

from functional_v1.common import ROOT,read,sha,stamp,write
from functional_v26.rank import fit,vector
from .intent import explicit_shadow_strength
from .prepare import DATA,EXP,PLAN,source_inputs

ORDER = {'complete':2,'partial':1,'failed':0}
MODELS = {'unconditional_pixels':(False,11),
          'conditional_parameters':(True,2),
          'conditional_pixels':(True,11)}


def bound(path,digest):
    if sha(path)!=digest:
        raise ValueError(f'Frozen file changed: {path}')
    return path


def case_book(mode):
    plan,old,held = source_inputs()
    review_path = EXP/f'{mode}-review.json'
    notes = read(review_path)
    bound(PLAN,notes['plan_sha256'])
    review = {row['id']:row for row in notes['cases']}
    cohort = 'previous_render_fit' if mode=='fit' else 'new_source_test'
    held_sources = {r['id']:r for r in held}
    if mode=='held':
        iso_path = EXP/'isolation.json'
        iso = read(bound(iso_path,notes['isolation_sha256']))
        bound(PLAN,iso['plan_sha256'])
        if iso['flags']:
            raise ValueError('Held source isolation failed')
        replay = read(bound(EXP/'held/replay.json',notes['replay_sha256']))
        bound(PLAN,replay['plan_sha256'])
        renders = {r['id']:r for r in replay['cases']}
    cases = []
    for case in plan['cases']:
        if case['cohort'] != cohort:
            continue
        cid = case['id']
        note = review[cid]
        for path,digest in note['panels'].items():
            bound(ROOT/path,digest)
        if mode=='fit':
            source = old[cid]
            fields = {key:source[key] for key in ('image','image_sha256','group_id')}
            rendered = [v for v in source['candidates'] if v['id'].startswith('initial-')]
        else:
            original = held_sources[cid]
            digest = iso['source_hashes'][cid]
            r = renders[cid]
            if digest != r['source_sha256']:
                raise ValueError(f'Wrong source rendered: {cid}')
            src = bound(ROOT/original['source'],digest)
            dst = DATA/'held-inputs'/(cid+src.suffix)
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,dst)
            bound(dst,digest)
            fields = dict(image=str(dst.relative_to(ROOT)),image_sha256=digest,group_id=original['group_id'])
            rendered = r['candidates'][1:]
        variants = []
        for i,params in enumerate(plan['initial_candidate_parameters'],1):
            variant = next(v for v in rendered if v['decision']['parameters']==params)
            bound(ROOT/variant['path'],variant['sha256'])
            variants.append(dict(id=f'v{i}',path=variant['path'],sha256=variant['sha256'],decision=variant['decision']))
        if set(note['labels'])!={'mild','strong'} or any(
            len(row)!=len(variants) or set(row)-{*ORDER,'uncertain'} for row in note['labels'].values()):
            raise ValueError(f'Invalid visual labels: {cid}')
        cases.append(dict(case,**fields,variants=variants,labels=note['labels'],evidence=note['evidence'],
                          panels=note['panels']))
    if len(cases)!=len(review):
        raise ValueError('Reviewed case count differs from plan')
    return cases,review_path


def interaction(x,intent,k):
    a = x[:k]
    return np.concatenate((a,intent*a))


def fit_stage():
    if (EXP/'fit.json').exists():
        raise FileExistsError('Keep the initial fit frozen')
    cases,review = case_book('fit')
    vectors,pairs = {},[]
    for case in cases:
        raw = {v['id']:vector(case,v) for v in case['variants']}
        for name,requests in case['requests'].items():
            levels = {explicit_shadow_strength(s) for s in requests.values()}
            if levels != ({-1} if name=='mild' else {1}):
                raise ValueError(f"Fit prompt does not give one matching strength: {case['id']}/{name}")
            intent = levels.pop()
            cid = f"{case['id']}/{name}"
            vectors[cid] = raw,intent
            votes = [(v,ORDER[y]) for v,y in zip(case['variants'],case['labels'][name]) if y in ORDER]
            for (a,va),(b,vb) in combinations(votes,2):
                if va==vb:
                    continue
                chosen,rejected = (a,b) if va>vb else (b,a)
                pairs.append(dict(case_id=cid,group_id=case['group_id'],chosen=chosen,rejected=rejected))
    weights = {}
    for name,(conditional,k) in MODELS.items():
        x = {cid:{vid:interaction(a,intent,k) if conditional else a[:k] for vid,a in raw.items()}
             for cid,(raw,intent) in vectors.items()}
        weights[name] = fit(pairs,x,k*2 if conditional else k).tolist()
    write(DATA/'fit-cases.json',cases)
    write(EXP/'fit.json',dict(at=stamp(),plan_sha256=sha(PLAN),fit_review_sha256=sha(review),
        fit_cases_sha256=sha(DATA/'fit-cases.json'),group_ids=[c['group_id'] for c in cases],
        pairs=len(pairs),weights=weights,GPU_hours=0))
    print(f'Fit {len(cases)} source groups, {len(pairs)} ordinal comparisons',flush=True)


def choose(vector_list,weights,conditional,k,intent):
    if conditional and intent is None:
        return None
    scores = [float((interaction(v,intent,k) if conditional else v[:k])@weights) for v in vector_list]
    # Candidate order has the mild setting first, for deterministic ties.
    return int(np.argmax(scores))


def summarize(predictions,case_book):
    result = {}
    cases = {c['id']:c for c in case_book}
    for name in predictions[0]['choices']:
        review = Counter(p['choices'][name]['verdict'] for p in predictions)
        grouped = {}
        for p in predictions:
            key = p['id'],p['request']
            grouped.setdefault(key,[]).append(p['choices'][name])
        good = {key:all(a['verdict']=='complete' for a in answers) for key,answers in grouped.items()}
        language_agreement = sum(len({a['choice'] for a in answers})==1
                                 and answers[0]['choice'] is not None for answers in grouped.values())
        reversals = 0
        for cid,c in cases.items():
            mild,strong = ({i for i,y in enumerate(c['labels'][s]) if y=='complete'}
                           for s in ('mild','strong'))
            if not (mild-strong and strong-mild and good[cid,'mild'] and good[cid,'strong']):
                continue
            choices = [{x['choice'] for x in grouped[cid,s]} for s in ('mild','strong')]
            reversals += len(choices[0])==len(choices[1])==1 and choices[0]!=choices[1]
        result[name] = dict(language_rows=dict(review),bilingual_completed_semantics=sum(good.values()),
            bilingual_same_nonempty_choice=language_agreement,
            both_requests_bilingually_completed=sum(good[cid,'mild'] and good[cid,'strong'] for cid in cases),
            successful_exclusive_strength_reversals=reversals)
    return result


def evaluation_stage():
    fit_path = EXP/'fit.json'
    trained = read(fit_path)
    bound(fit_path,read(EXP/'held-review.json')['fit_preceding_review_sha256'])
    bound(PLAN,trained['plan_sha256'])
    bound(EXP/'fit-review.json',trained['fit_review_sha256'])
    bound(DATA/'fit-cases.json',trained['fit_cases_sha256'])
    cases,review = case_book('held')
    if set(trained['group_ids']) & {c['group_id'] for c in cases}:
        raise ValueError('Source group leaked into held set')
    rows = []
    for case in cases:
        values = [vector(case,v) for v in case['variants']]
        for name,requests in case['requests'].items():
            for lang,request in requests.items():
                intent = explicit_shadow_strength(request)
                choices = {'fixed_mild':0,'parsed_preset':None if intent is None else int(intent>0)}
                for model,(conditional,k) in MODELS.items():
                    choices[model] = choose(values,np.array(trained['weights'][model]),conditional,k,intent)
                rows.append(dict(id=case['id'],domain=case['domain'],request=name,language=lang,
                    instruction=request,parsed_intent=intent,
                    choices={m:dict(choice=None if i is None else case['variants'][i]['id'],
                         verdict='abstained' if i is None else case['labels'][name][i]) for m,i in choices.items()}))
    feasible = [sum('complete' in c['labels'][name] for c in cases) for name in ('mild','strong')]
    pairs = [{i for i,y in enumerate(c['labels'][name]) if y=='complete'}
             for c in cases for name in ('mild','strong')]
    exclusive = sum(bool(m-s and s-m) for m,s in zip(pairs[::2],pairs[1::2]))
    write(DATA/'held-cases.json',cases)
    write(EXP/'result.json',dict(at=stamp(),plan_sha256=sha(PLAN),fit_sha256=sha(fit_path),
        held_review_sha256=sha(review),held_cases_sha256=sha(DATA/'held-cases.json'),
        fit_groups=len(trained['group_ids']),held_groups=len(cases),
        candidate_ceiling=dict(mild_feasible=feasible[0],strong_feasible=feasible[1],exclusive_reversal_groups=exclusive),
        literal_language_records=len(rows),summary=summarize(rows,cases),predictions=rows,
        GPU_hours=0,deployed=False,final_acceptance_opened=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=('fit','evaluate'))
    args = parser.parse_args()
    if args.stage=='fit':
        fit_stage()
    else:
        evaluation_stage()


if __name__ == '__main__':
    main()

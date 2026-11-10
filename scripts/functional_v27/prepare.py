"""Render and isolate the source-disjoint R27 evaluation pool."""
import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

from functional_v1.common import ROOT,read,sha,stamp,write
from functional_v24.probe import protected_sources,render,SourceMatcher
from functional_v26.shortlist import SCAN

EXP = ROOT/'experiments/functional-v27-20261107'
PRIOR = ROOT/'data/training/functional-v26-20261107'
DATA = ROOT/'data/training/functional-v27-20261107'
PLAN = EXP/'plan.json'


def source_inputs():
    plan = read(PLAN)
    paths = [(SCAN,plan['scan_sha256']),
             (PRIOR/'reviewed-candidates.json',plan['prior_cases_sha256']),
             (Path(__file__).with_name('intent.py'),plan['intent_code_sha256'])]
    for path,digest in paths:
        if sha(path)!=digest:
            raise ValueError(f'Frozen source changed: {path}')
    old = {r['id']:r for r in read(PRIOR/'reviewed-candidates.json')}
    scan = {r['id']:r for r in read(SCAN)['rows']}
    held = [dict(scan[s['id']],crop_center_fraction=s['crop_center_fraction'],
                 instructions={'en':f"Two strengths | {s['focus']['en']}"},
                 edits=plan['initial_candidate_parameters'])
            for s in plan['cases'] if s['cohort']=='new_source_test']
    if len({r['group_id'] for r in held}) != len(held) or (
        {r['group_id'] for r in held} & {r['group_id'] for r in old.values()}):
        raise ValueError('Repeated source groups in the new holdout')
    return plan,old,held


def isolate(old,held):
    protected = protected_sources()
    r24 = ROOT/'data/training/functional-v24-20261107/reviewed-candidates.json'
    for source in read(r24)+list(old.values()):
        path = ROOT/source['source']
        if sha(path)!=source['image_sha256']:
            raise ValueError(f"Prior source changed: {source['id']}")
        protected[str(path)] = 'prior_stored_sources'
    missing = {p:kind for p,kind in protected.items() if not Path(p).is_file()}
    if set(missing.values()) - {'GIER_test','GIER_val'}:
        raise ValueError('An internal protected source is missing')
    matcher,flags = SourceMatcher(),[]
    for source in held:
        for path,kind in protected.items():
            if path in missing:
                continue
            match = matcher.compare(ROOT/source['source'],path)
            if match:
                flags.append(dict(case=source['id'],other=path,split=kind,evidence=match))
        print(f"Checked {source['id']}",flush=True)
    for a,b in combinations(held,2):
        match = matcher.compare(ROOT/a['source'],ROOT/b['source'])
        if match:
            flags.append(dict(case=a['id'],other_case=b['id'],evidence=match))
    write(EXP/'isolation.json',dict(at=stamp(),plan_sha256=sha(PLAN),flags=flags,
        source_hashes={r['id']:sha(ROOT/r['source']) for r in held},
        protected_paths_by_split=dict(Counter(protected.values())),
        missing=missing,comparisons=len(matcher.pairs),GPU_hours=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=('render','isolate'))
    args = parser.parse_args()
    _,old,held = source_inputs()
    if args.stage=='render':
        render(held,PLAN,EXP/'held')
    else:
        isolate(old,held)


if __name__=='__main__':
    main()

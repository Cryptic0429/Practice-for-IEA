"""Render paired supervision candidates and check source isolation."""
import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

from functional_v1.common import ROOT,read,sha,stamp,write
from functional_v24.probe import protected_sources,render,SourceMatcher
from .shortlist import EXP,SCAN,R24


def selected():
    path = EXP/'plan.json'
    plan = read(path)
    if plan['scan_sha256'] != sha(SCAN):
        raise ValueError('Original scan changed')
    for name, digest in plan['preview_lists'].items():
        if sha(EXP/name) != digest:
            raise ValueError(f'Preview list changed: {name}')
    rows = {r['id']:r for r in read(SCAN)['rows']}
    cases = [dict(rows[c['id']], **{k:v for k,v in c.items() if k != 'id'}) for c in plan['cases']]
    if len({c['group_id'] for c in cases}) != len(cases):
        raise ValueError('Repeated source group')
    return path,cases


def isolate(cases, plan_path):
    protected = protected_sources()
    r24 = ROOT/'data/training/functional-v24-20261107/reviewed-candidates.json'
    r24_isolation = read(R24/'isolation.json')
    for c in read(r24):
        path = ROOT/c['source']
        if sha(path) != r24_isolation['candidate_source_hashes'][c['id']]:
            raise ValueError(f'R24 source mismatch: {c["id"]}')
        protected[str(path)] = 'R24_reviewed'
    missing = {p:kind for p,kind in protected.items() if not Path(p).is_file()}
    if set(missing.values()) - {'GIER_test','GIER_val'}:
        raise ValueError('An internal protected source is missing')
    matcher = SourceMatcher()
    flags = []
    for c in cases:
        for path,kind in protected.items():
            if path in missing:
                continue
            match = matcher.compare(ROOT/c['source'],path)
            if match:
                flags.append(dict(case=c['id'], protected=path, split=kind, evidence=match))
        print(f"Checked source {c['id']}",flush=True)
    for a,b in combinations(cases,2):
        match = matcher.compare(ROOT/a['source'], ROOT/b['source'])
        if match:
            flags.append(dict(case=a['id'], other_candidate=b['id'], evidence=match))
    write(EXP/'isolation.json',dict(at=stamp(),plan_sha256=sha(plan_path),flags=flags,
        protected_paths_by_split=dict(Counter(protected.values())),missing=missing,
        candidate_source_hashes={r['id']:sha(ROOT/r['source']) for r in cases},
        comparisons=len(matcher.pairs),GPU_hours=0,
        method='R24 pHash/gray correlation and ORB matcher; protected source pixels only. Exact source groups checked before comparison.'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=('render','render-refinement','isolate'))
    args = parser.parse_args()
    plan_path,cases = selected()
    if args.stage == 'render':
        render(cases,plan_path,EXP)
    elif args.stage == 'render-refinement':
        plan_path = EXP/'refinement-plan.json'
        plan = read(plan_path)
        if sha(EXP/'replay.json') != plan['initial_replay_sha256']:
            raise ValueError('Initial replay changed')
        by_id = {c['id']:c for c in cases}
        refined = []
        for spec in plan['cases']:
            initial = by_id[spec['id']]
            if spec['instructions'] != initial['instructions']:
                raise ValueError('Refinement changed the literal request')
            refined.append(dict(initial,edits=spec['edits'],crop_center_fraction=spec['crop_center_fraction']))
        render(refined,plan_path,EXP/'refinements')
    else:
        isolate(cases,plan_path)


if __name__ == '__main__':
    main()

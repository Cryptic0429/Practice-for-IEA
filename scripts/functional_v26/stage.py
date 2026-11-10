"""Stage visually verified same-source preference candidates without training admission."""
from collections import Counter
import shutil

import numpy as np

from functional_v1.common import ROOT,read,sha,stamp,write
from functional_v15.editor import load_rgb
from functional_v15.outcomes import execute
from .replay import selected
from .shortlist import EXP

DATA = ROOT/'data/training/functional-v26-20261107'


def bound(path,digest):
    if sha(path) != digest:
        raise ValueError(f'File changed: {path}')
    return path


def compare(case, candidates):
    for chosen in candidates:
        if chosen['verdict'] != 'complete':
            continue
        for rejected in candidates:
            if rejected['verdict'] == 'failed':
                yield dict(id=f"{case['id']}/{chosen['id']}>{rejected['id']}",case_id=case['id'],
                    group_id=case['group_id'],domain=case['domain'],image=case['image'],
                    image_sha256=case['image_sha256'],instructions=case['instructions'],
                    chosen=chosen,rejected=rejected)


def main():
    plan,selected_rows = selected()
    iso_path = EXP/'isolation.json'
    iso = read(iso_path)
    if iso['flags'] or iso['plan_sha256'] != sha(plan):
        raise ValueError('Candidate source isolation failed')
    review_path = EXP/'visual-review.json'
    review = read(review_path)
    bound(EXP/'detail.json',review['detail_sha256'])
    note_by_case = {c['id']:c for c in review['cases']}
    rendered = {}
    for stage, root, plan_name, digest_key in (
        ('initial',EXP,'plan.json','replay_sha256'),
        ('refined',EXP/'refinements','refinement-plan.json','refined_replay_sha256'),
    ):
        manifest = root/'replay.json'
        replay = read(bound(manifest,review[digest_key]))
        bound(EXP/plan_name,replay['plan_sha256'])
        rendered[stage] = {r['id']:r for r in replay['cases']}
    data_rows, pairs = [], []
    for row in selected_rows:
        cid = row['id']
        source_digest = iso['candidate_source_hashes'][cid]
        source = bound(ROOT/row['source'],source_digest)
        notes = note_by_case[cid]
        for panel,digest in notes['panels'].items():
            bound(EXP/panel,digest)
        img = DATA/'inputs'/(cid+source.suffix)
        img.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,img)
        bound(img,source_digest)
        rgb = load_rgb(img)
        case = dict(id=cid,group_id=row['group_id'],domain=row['domain'],
                    image=str(img.relative_to(ROOT)),image_sha256=source_digest,
                    source=row['source'],source_annotation_indices=row['annotation_indices'],
                    instructions=row['instructions'],instruction_origin='AI-authored paired translations',
                    panels=notes['panels'])
        candidates = []
        for stage, cohort in rendered.items():
            if cid not in cohort:
                continue
            r = cohort[cid]
            if r['source_sha256'] != source_digest:
                raise ValueError(f'Replay source differs: {cid}')
            assessment = {a['index']:a for a in notes['candidates' if stage=='initial' else 'refined_candidates']}
            if set(assessment) != {v['index'] for v in r['candidates'] if v['index'] != 0}:
                raise ValueError(f'Incomplete assessment: {stage}/{cid}')
            for v in r['candidates']:
                path = bound(ROOT/v['path'],v['sha256'])
                if v['index'] == 0:
                    if not np.array_equal(load_rgb(path),rgb):
                        raise ValueError(f'Original pixel mismatch: {stage}/{cid}')
                    continue
                note = assessment[v['index']]
                if note['verdict']=='complete' and not np.array_equal(
                    execute(rgb,v['decision'])['result'],load_rgb(path)):
                    raise ValueError(f'Chosen output not reproducible: {stage}/{cid}/{v["index"]}')
                candidates.append(dict(v,**{k:value for k,value in note.items() if k!='index'},
                                       id=f"{stage}-{v['index']}"))
        case['candidates'] = candidates
        data_rows.append(case)
        pairs.extend(compare(case,candidates))
    write(DATA/'reviewed-candidates.json',data_rows)
    write(DATA/'preference-candidates.json',pairs)
    counts = Counter(v['verdict'] for c in data_rows for v in c['candidates'])
    pair_groups = {r['group_id']:r['domain'] for r in pairs}
    write(DATA/'receipt.json',dict(at=stamp(),status='staged_single_AI_reviewer_no_training_admission',
          review_sha256=sha(review_path),isolation_sha256=sha(iso_path),
          cases_sha256=sha(DATA/'reviewed-candidates.json'),pairs_sha256=sha(DATA/'preference-candidates.json'),
          candidates_by_verdict=dict(counts),candidate_groups=len(data_rows),pair_records=len(pairs),
          paired_original_groups=len(pair_groups),paired_domains=dict(Counter(pair_groups.values())),
          pair_rejection_causes=dict(Counter(p['rejected']['cause'] for p in pairs)),
          independent_Qwen_review=False,independent_user_review=False,GPU_hours=0,
          final_acceptance_opened=False,training_admitted=False))
    print(f'Verified {len(data_rows)} source groups, {sum(counts.values())} edits, {len(pairs)} preferences')


if __name__ == '__main__':
    main()

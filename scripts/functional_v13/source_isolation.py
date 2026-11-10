"""CPU-only original/near-duplicate isolation, never a visual quality score."""
from itertools import combinations
from collections import Counter
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v6.isolation import SourceMatcher
from .source_candidates import EXP


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--extended',action='store_true');args=ap.parse_args()
    dest=EXP/('source-isolation-extended.json' if args.extended else 'source-isolation.json');assert not dest.exists()
    mf=EXP/('source-review-extended' if args.extended else 'source-review')/'manifest.json';sources=read(mf)['candidates']
    assignment=ROOT/'data/stage1/splits/image-edit-3k-v1/group-assignments.json'
    groups=read(assignment);by_path={p:g for g in groups for p in g['source_paths']}
    held={p:g['group_id'] for g in groups if g['partition']!='training_pool' for p in g['source_paths']}
    metadata_files={str(assignment.relative_to(ROOT)):sha(assignment)}
    # Read only source identity fields, not acceptance instructions or targets.
    protocol=ROOT/'experiments/functional-v12-20261106/evaluation-protocol.json'
    for spec in read(protocol)['case_files'].values():
        path=ROOT/spec['path'];assert sha(path)==spec['sha256'];metadata_files[spec['path']]=sha(path)
        for c in read(path):held[str(ROOT/c['image'])]=c['group_id']
    prior=ROOT/'data/training/functional-v11-20261106/cases.json';parents={str(ROOT/c['image']):c['group_id'] for c in read(prior)}
    metadata_files[str(prior.relative_to(ROOT))]=sha(prior)
    matcher=SourceMatcher();flags=[]
    for c in sources:
        assert by_path[c['source_image']]['partition']=='training_pool'
        assert sha(c['source_image'])==c['source_sha256']
        for kind,others in [('reserved_or_prior_holdout',held),('already_parent_training',parents)]:
            for path,gid in others.items():
                e=matcher.compare(c['source_image'],path)
                if c['group_id']==gid or e:flags.append(dict(source=c['id'],other_group=gid,other_image=path,kind=kind,evidence=e))
        write(EXP/'source-isolation-progress.json',dict(at=stamp(),source=c['id'],pairs=len(matcher.pairs),flags=len(flags),GPU_hours=0))
    for a,b in combinations(sources,2):
        e=matcher.compare(a['source_image'],b['source_image'])
        if a['group_id']==b['group_id'] or e:flags.append(dict(source=a['id'],other_source=b['id'],kind='within_new_candidates',evidence=e))
    excluded={r['source'] for r in flags if r['kind']!='within_new_candidates'}
    for r in flags:
        if r['kind']=='within_new_candidates' and r['source'] not in excluded:excluded.add(r['other_source'])
    remaining=[c for c in sources if c['id'] not in excluded]
    write(dest,dict(at=stamp(),manifest_sha256=sha(mf),metadata_files=metadata_files,flags=flags,excluded=sorted(excluded),
        candidate_counts=dict(Counter(c['domain'] for c in sources)),remaining_counts=dict(Counter(c['domain'] for c in remaining)),
        remaining_ids=[c['id'] for c in remaining],heldout_unique_input_paths=len(held),parent_unique_inputs=len(parents),
        compared_pairs=len(matcher.pairs),GPU_hours=0,training_started=False,
        method='Original connected-component groups plus inherited conservative pHash/gray correlation/ORB homography matcher. All reserved source fingerprints used only for isolation; no heldout output, instruction or quality review. Related views may still escape detection.',
        policy='Exclude every reserved/parent collision and one side of within-candidate flags before admission. Do not count near-parent photographs as newly covered groups.'))
    print('Isolation remaining:',dict(Counter(c['domain'] for c in remaining)),'flags',len(flags))


if __name__=='__main__':main()

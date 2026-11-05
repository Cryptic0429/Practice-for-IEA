"""Pixel/perceptual source-isolation audit only. Never assess editing quality with these numbers."""
import json
from itertools import combinations
from collections import Counter
import cv2
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.prepare import fingerprint, near
from functional_v1.editor import load_rgb
from .source_bank import EXP


def main():
    cv2.setNumThreads(2)
    manifest = read(EXP/'source-review/manifest.json')
    observations = list(map(json.loads, (EXP/'source-review/observations.jsonl').read_text().splitlines()))
    notes = {r['id']: r for r in observations}
    assert len(notes) == len(observations)
    byid = {c['id']: c for c in manifest['candidates']}
    for r in observations:
        assert r['source_sha256'] == byid[r['id']]['source_sha256']
        assert sha(ROOT/r['sheet']) == r['sheet_sha256']
    features = {}
    def get(path):
        path = str(path)
        if path not in features:
            features[path] = fingerprint(load_rgb(path))
        return features[path]
    old = read(ROOT/'data/training/quality-v2-20261103/train3k.json')
    oldpaths = {r['images'][0]: r['group_id'] for r in old}
    historical = [(gid, get(path)) for path, gid in oldpaths.items()]
    held = []
    for directory, names in [('functional-v1-20261104', ('development-cases.json', 'acceptance-cases.json')),
                              ('functional-v2-20261104', ('development-final-cases.json', 'acceptance-final-cases.json'))]:
        for name in names:
            for c in read(ROOT/'data/training'/directory/name):
                path = ROOT/c['image']
                if str(path) not in {r[0] for r in held}:
                    held.append((str(path), c['group_id'], get(path)))
    collisions = []
    for c in manifest['candidates']:
        assert sha(c['source_image']) == c['source_sha256']
        f = get(c['source_image'])
        if c['split'] != 'train':
            for gid, other in historical:
                if c['group_id'] == gid or near(f, other):
                    collisions.append(dict(a=c['id'], b=gid, reason='candidate_development_near_historical_training'))
        else:
            for path, gid, other in held:
                if c['group_id'] == gid or near(f, other):
                    collisions.append(dict(a=c['id'], b=gid, reason='candidate_training_near_prior_holdout'))
    for a, b in combinations(manifest['candidates'], 2):
        if a['split'] == b['split'] == 'train':
            continue  # Intra-training near duplicates will be checked on the chosen source list too.
        if a['group_id'] == b['group_id'] or near(get(a['source_image']), get(b['source_image'])):
            collisions.append(dict(a=a['id'], b=b['id'], reason='near_current_sources', cross_split=a['split'] != b['split']))
    output = EXP/'planning/source-candidate-audit.json'
    write(output, dict(at=stamp(), manifest_sha256=sha(EXP/'source-review/manifest.json'),
          observations_sha256=sha(EXP/'source-review/observations.jsonl'), candidate_sources=len(byid),
          observed_sources=len(notes), inherited_training_groups=len({r['group_id'] for r in old}),
          inherited_training_paths=len(oldpaths), prior_holdout_input_paths=len(held),
          collisions=collisions, counts=dict(Counter(c['reason'] for c in collisions)),
          method='Original connected-component group IDs, plus pHash Hamming<=6 and grayscale correlation>.985 or MAE<.01. No claim to detect all crops or related shoots.',
          status='candidate_audit_not_final_source_freeze', labels_admitted=False,
          rule='Historical-training collisions cannot enter development. Prior-holdout collisions cannot enter training. Chosen source sets must be mutually disjoint after resolving cross-split and within-development collisions. No outcome or reference inspected here.'))
    print(json.dumps({k: v for k, v in read(output).items() if k not in ('collisions',)}))


if __name__ == '__main__':
    main()

"""Additional crop/geometry duplicate screening, motivated by a directly observed missed pair."""
import json
from itertools import combinations
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.prepare import fingerprint
from functional_v1.editor import load_rgb
from .source_bank import EXP


def main():
    cv2.setNumThreads(2)
    manifest = read(EXP/'source-review/manifest.json')
    candidates = manifest['candidates']
    cache = {}
    orb = cv2.ORB_create(nfeatures=1000)
    def get(path):
        path = str(path)
        if path not in cache:
            rgb = load_rgb(path)
            g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            scale = min(1., 640/max(g.shape))
            g = cv2.resize(g, (max(1, round(g.shape[1]*scale)), max(1, round(g.shape[0]*scale))))
            k, d = orb.detectAndCompute(g, None)
            cache[path] = dict(hash=fingerprint(rgb)[0], points=np.array([p.pt for p in k], dtype=np.float32),
                               descriptors=d, area=g.size)
        return cache[path]
    checked = {}
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    def compare(a, b):
        pair = tuple(sorted([str(a), str(b)]))
        if pair in checked: return checked[pair]
        x, y = get(pair[0]), get(pair[1])
        distance = (x['hash']^y['hash']).bit_count()
        result = None
        if distance <= 20 and x['descriptors'] is not None and y['descriptors'] is not None and min(len(x['descriptors']), len(y['descriptors'])) >= 20:
            matches = matcher.knnMatch(x['descriptors'], y['descriptors'], k=2)
            good = [m[0] for m in matches if len(m) == 2 and m[0].distance < .75*m[1].distance]
            if len(good) >= 20:
                p = np.float32([x['points'][m.queryIdx] for m in good])
                q = np.float32([y['points'][m.trainIdx] for m in good])
                cv2.setRNGSeed(20261105)
                h, inside = cv2.findHomography(p, q, cv2.RANSAC, 4)
                if inside is not None and int(inside.sum()) >= 15 and float(inside.mean()) >= .6:
                    active = inside.ravel().astype(bool)
                    coverage = [float(cv2.contourArea(cv2.convexHull(v[active])))/z['area'] for v, z in ((p, x), (q, y))]
                    if min(coverage) >= .03:
                        result = dict(hash_hamming=distance, ratio_matches=len(good), ransac_inliers=int(inside.sum()),
                                      inlier_fraction=float(inside.mean()), hull_coverage=coverage)
        checked[pair] = result
        return result
    byid = {r['id']: r for r in candidates}
    known = ('development-native-056', 'development-native-151')
    proof = compare(byid[known[0]]['source_image'], byid[known[1]]['source_image'])
    assert proof is not None, 'Directly observed missed crop/related-shot pair must be detected'
    write(EXP/'planning/known-visual-near-pair.json', dict(at=stamp(), ids=list(known), geometry=proof,
          source_sha256=[byid[k]['source_sha256'] for k in known],
          evidence='Codex directly viewed both source sheets: same woman, clothing, porch and very similar framing. Existing pHash<=6 screen missed it (distance16). Keep as one near-source cluster or exclude one; geometry supports shared image content.',
          editing_quality_claim=False))
    old = read(ROOT/'data/training/quality-v2-20261103/train3k.json')
    historical = {r['images'][0]: r['group_id'] for r in old}
    held = {}
    for directory, names in [('functional-v1-20261104', ('development-cases.json', 'acceptance-cases.json')),
                              ('functional-v2-20261104', ('development-final-cases.json', 'acceptance-final-cases.json'))]:
        for name in names:
            for c in read(ROOT/'data/training'/directory/name): held[str(ROOT/c['image'])] = c['group_id']
    collisions = []
    for c in candidates:
        others = historical if c['split'] != 'train' else held
        reason = 'development_near_historical_training' if c['split'] != 'train' else 'training_near_prior_holdout'
        for path, gid in others.items():
            evidence = compare(c['source_image'], path)
            if evidence is not None:
                collisions.append(dict(a=c['id'], b=gid, other_image=path, reason=reason, geometry=evidence))
        write(EXP/'planning/source-orb-status.json', dict(status='running', candidate=c['id'], compared_unique_pairs=len(checked), at=stamp()))
    for a, b in combinations(candidates, 2):
        if a['split'] == b['split'] == 'train': continue
        evidence = compare(a['source_image'], b['source_image'])
        if evidence is not None:
            collisions.append(dict(a=a['id'], b=b['id'], reason='near_current_sources',
                                   cross_split=a['split'] != b['split'], geometry=evidence))
    write(EXP/'planning/source-geometry-audit.json', dict(at=stamp(), status='candidate_geometry_screen_not_final_freeze',
        manifest_sha256=sha(EXP/'source-review/manifest.json'), collisions=collisions,
        counts=dict(Counter(c['reason'] for c in collisions)), unique_pairs=len(checked),
        method='Coarse pHash<=20; ORB1000 at maxedge640; Hamming ratio<.75, >=20 matches; RANSAC4px >=15inliers, fraction>=.6, convexhull coverage>=3% in both photos. Potential shared image content/related shots, not a functional-quality score.',
        rule='Resolve or exclude every selected cross-split candidate flag before freezing. Preserve manual related-source clusters. These checks can still miss other crops or shoots.',
        known_directly_viewed_missed_pair=known, inference_or_training_started=False))
    write(EXP/'planning/source-orb-status.json', dict(status='completed', at=stamp(), counts=dict(Counter(c['reason'] for c in collisions))))
    print('geometry source screen complete', Counter(c['reason'] for c in collisions))


if __name__ == '__main__':
    main()

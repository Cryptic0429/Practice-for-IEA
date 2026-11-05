"""Auditable source grouping plus conservative perceptual and geometry duplicate flags."""
import json
from collections import Counter
from itertools import combinations
import cv2
import numpy as np
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.editor import load_rgb
from functional_v1.prepare import fingerprint, near
from .source_bank import EXP, DATA


class SourceMatcher:
    def __init__(self):
        cv2.setNumThreads(2)
        self.cache = {}
        self.pairs = {}
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def get(self, path):
        path = str(path)
        if path not in self.cache:
            rgb = load_rgb(path)
            g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            scale = min(1., 640/max(g.shape))
            g = cv2.resize(g, (max(1, round(g.shape[1]*scale)), max(1, round(g.shape[0]*scale))))
            k, d = self.orb.detectAndCompute(g, None)
            self.cache[path] = dict(fingerprint=fingerprint(rgb), descriptors=d, area=g.size,
                                   points=np.array([p.pt for p in k], dtype=np.float32))
        return self.cache[path]

    def compare(self, a, b):
        pair = tuple(sorted([str(a), str(b)]))
        if pair in self.pairs: return self.pairs[pair]
        x, y = self.get(pair[0]), self.get(pair[1])
        result = None
        if near(x['fingerprint'], y['fingerprint']):
            result = dict(method='pHash_gray_correlation_or_MAE')
        distance = (x['fingerprint'][0]^y['fingerprint'][0]).bit_count()
        if distance <= 20 and x['descriptors'] is not None and y['descriptors'] is not None and min(len(x['descriptors']), len(y['descriptors'])) >= 20:
            matches = self.matcher.knnMatch(x['descriptors'], y['descriptors'], k=2)
            good = [m[0] for m in matches if len(m) == 2 and m[0].distance < .75*m[1].distance]
            if len(good) >= 20:
                p = np.float32([x['points'][m.queryIdx] for m in good])
                q = np.float32([y['points'][m.trainIdx] for m in good])
                cv2.setRNGSeed(20261105)
                _, inside = cv2.findHomography(p, q, cv2.RANSAC, 4)
                if inside is not None and int(inside.sum()) >= 15 and float(inside.mean()) >= .6:
                    active = inside.ravel().astype(bool)
                    coverage = [float(cv2.contourArea(cv2.convexHull(v[active])))/z['area'] for v, z in ((p, x), (q, y))]
                    if min(coverage) >= .03:
                        result = dict(method='ORB_geometry', hash_hamming=distance, ratio_matches=len(good),
                                      ransac_inliers=int(inside.sum()), inlier_fraction=float(inside.mean()), hull_coverage=coverage)
        self.pairs[pair] = result
        return result


def observed_bank():
    sources, notes, files = {}, {}, []
    for folder in (EXP/'source-review', EXP/'source-extension/train', EXP/'source-extension/acceptance'):
        manifest, obs = folder/'manifest.json', folder/'observations.jsonl'
        files.extend((manifest, obs))
        sources.update((r['id'], r) for r in read(manifest)['candidates'])
        for r in map(json.loads, obs.read_text().splitlines()):
            assert r['id'] not in notes
            assert sources[r['id']]['source_sha256'] == r['source_sha256']
            assert sha(ROOT/r['sheet']) == r['sheet_sha256']
            notes[r['id']] = r
    return sources, notes, files


def main():
    output = EXP/'planning/observed-source-isolation.json'
    assert not output.exists(), 'Keep pre-admission source audit immutable'
    bank, notes, files = observed_bank()
    current = [dict(bank[sid], domain=n['domain']) for sid, n in notes.items()
               if n['source_usable'] and n['request_scope'] == 'global' and n['domain'] in ('portrait', 'landscape')
               and (sid.startswith('train-') or sid.startswith('acceptance-'))]
    dev = read(DATA/'development-native-cases.json')
    current.extend(dict(id=c['id'], group_id=c['group_id'], source_image=str(ROOT/c['image']), split='development', domain=c['domain']) for c in dev)
    old = read(ROOT/'data/training/quality-v2-20261103/train3k.json')
    historical = {r['images'][0]: r['group_id'] for r in old}
    for c in read(ROOT/'data/training/functional-v4-20261105/cases.json'):
        historical[str(ROOT/c['image'])] = c['group_id']
    held = {}
    for directory, names in [('functional-v1-20261104', ('development-cases.json', 'acceptance-cases.json')),
                              ('functional-v2-20261104', ('development-final-cases.json', 'acceptance-final-cases.json'))]:
        for name in names:
            for c in read(ROOT/'data/training'/directory/name): held[str(ROOT/c['image'])] = c['group_id']
    matcher = SourceMatcher()
    collisions = []
    for c in current:
        if 'source_sha256' in c: assert sha(c['source_image']) == c['source_sha256']
        others = held if c['split'] == 'train' else historical
        reason = 'training_near_prior_holdout' if c['split'] == 'train' else 'holdout_near_historical_training'
        for path, gid in others.items():
            evidence = matcher.compare(c['source_image'], path)
            if gid == c['group_id'] or evidence:
                collisions.append(dict(a=c['id'], b=gid, other_image=path, reason=reason,
                                       group_identity=gid == c['group_id'], geometry=evidence))
        write(EXP/'planning/observed-isolation-status.json', dict(status='running', source=c['id'], at=stamp(),
              unique_image_pairs=len(matcher.pairs), flags=len(collisions)))
    for a, b in combinations(current, 2):
        evidence = matcher.compare(a['source_image'], b['source_image'])
        if a['group_id'] == b['group_id'] or evidence:
            collisions.append(dict(a=a['id'], b=b['id'], reason='near_current_sources', cross_split=a['split'] != b['split'],
                group_identity=a['group_id'] == b['group_id'], geometry=evidence))
    # Positive control: a visually confirmed related pair missed by old pHash<=6.
    proof = matcher.compare(bank['development-native-056']['source_image'], bank['development-native-151']['source_image'])
    assert proof and proof['method'] == 'ORB_geometry'
    write(output, dict(at=stamp(), status='observed_sources_screened_not_final_train_admission',
        source_files={str(p.relative_to(ROOT)): sha(p) for p in files}, development_cases_sha256=sha(DATA/'development-native-cases.json'),
        observed_candidate_counts=dict(Counter((c['split']+'_'+c['domain']) for c in current)),
        historic_training_input_paths=len(historical), historic_holdout_input_paths=len(held),
        candidates=[dict(id=c['id'], group_id=c['group_id'], split=c['split'], source_image=c['source_image']) for c in current],
        collisions=collisions, counts=dict(Counter(c['reason'] for c in collisions)), unique_pairs=len(matcher.pairs),
        positive_control=proof,
        method='Connected source groups; pHash<=6 with gray correlation>.985 or MAE<.01; additional ORB1000 maxedge640 screening at pHash<=20, ratio<.75 >=20matches, RANSAC4px >=15inliers and >=.6fraction, hull coverage>=3% both. Manual related-image notes retained. Can still miss other crops/related shoots; not a functional-quality metric.',
        labels_admitted=False, model_outputs_generated=False))
    write(EXP/'planning/observed-isolation-status.json', dict(status='completed', at=stamp(), flags=len(collisions)))
    print('Source isolation audit complete', Counter(c['reason'] for c in collisions))


if __name__ == '__main__': main()

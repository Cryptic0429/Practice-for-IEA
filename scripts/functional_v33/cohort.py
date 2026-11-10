"""Freeze the R33 contrast holdout before replaying the existing four recipes."""
import argparse
import hashlib
import shutil
from itertools import combinations
from pathlib import Path

import cv2
from PIL import Image

from functional_v1.common import ROOT, read, sha, stamp, write
from functional_v15.editor import load_rgb, save
from functional_v24.probe import SourceMatcher, protected_sources
from functional_v30.render import panel
from functional_v32.editor import RECIPES, apply

NAME = 'functional-v33-20261107'
EXP = ROOT / 'experiments' / NAME
DATA = ROOT / 'data/training' / NAME
PLAN = EXP / 'protocol.json'
R32 = ROOT / 'experiments/functional-v32-20261107'
R32_DATA = ROOT / 'data/training/functional-v32-20261107'


def isolate(photos):
    protected = protected_sources()
    for version, name in ((24, 'reviewed-candidates'), (26, 'reviewed-candidates'), (27, 'held-cases')):
        for row in read(ROOT / f'data/training/functional-v{version}-20261107/{name}.json'):
            protected[str(ROOT / row['image'])] = f'r{version}'
    for version in (29, 30, 31, 32):
        for row in read(ROOT / f'experiments/functional-v{version}-20261107/replay.json')['cases']:
            for key in ('source', 'path'):
                if key in row:
                    protected[str(ROOT / row[key])] = f'r{version}'
    missing = {p: kind for p, kind in protected.items() if not Path(p).is_file()}
    if set(missing.values()) - {'GIER_val', 'GIER_test'}:
        raise ValueError('Missing protected internal original')
    available = [p for p in protected if p not in missing]
    groups = {r['group_id'] for r in read(R32 / 'replay.json')['cases']}
    if groups.intersection(r['group_id'] for r in photos):
        raise ValueError('R32 source group appears in holdout')
    matcher = SourceMatcher()
    flags = []
    for row in photos:
        for other in available:
            match = matcher.compare(ROOT / row['path'], other)
            if match:
                flags.append(dict(id=row['id'], other=other, match=match))
        print('Compared', row['id'], flush=True)
    for a, b in combinations(photos, 2):
        match = matcher.compare(ROOT / a['path'], ROOT / b['path'])
        if match or a['group_id'] == b['group_id']:
            flags.append(dict(id=a['id'], other=b['id'], match=match))
    write(EXP / 'isolation.json', dict(at=stamp(), protocol_sha256=sha(PLAN),
          available_protected=len(available), missing=missing, flags=flags, comparisons=len(matcher.pairs),
          source_hashes={r['id']: sha(ROOT / r['path']) for r in photos}, GPU_hours=0))


def render(photos):
    proof = read(EXP / 'isolation.json')
    if proof['flags'] or proof['protocol_sha256'] != sha(PLAN):
        raise ValueError('Holdout isolation has not passed')
    rows = []
    for photo in photos:
        source = ROOT / photo['path']
        source_hash = sha(source)
        if source_hash != proof['source_hashes'][photo['id']]:
            raise ValueError('Original changed after isolation')
        copy = DATA / 'held-inputs' / (photo['id'] + source.suffix)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copy)
        original = load_rgb(copy)
        for direction, requests in photo['requests'].items():
            case_id = f"{photo['id']}-{direction}"
            order = sorted(RECIPES, key=lambda recipe: hashlib.sha256(f'r33/{case_id}/{recipe}'.encode()).digest())
            images, tiles = {}, []
            for recipe in order:
                edited = apply(original, direction, recipe)
                path = EXP / 'images' / f'{case_id}-{recipe}.png'
                save(path, edited)
                images[recipe] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path))
                tiles.append(Image.fromarray(edited))
            board, box = panel(Image.fromarray(original), tiles, photo['crop'])
            path = EXP / 'boards' / f'{case_id}.png'
            path.parent.mkdir(exist_ok=True)
            board.save(path)
            rows.append(dict(id=case_id, group_id=photo['group_id'], photo_id=photo['id'],
                             domain=photo['domain'], light=photo['light'], direction=direction, requests=requests,
                             candidate_order=order, source=str(copy.relative_to(ROOT)), source_sha256=source_hash,
                             images=images, board=str(path.relative_to(ROOT)), board_sha256=sha(path), native_crop=box))
        print('Rendered', photo['id'], flush=True)
    write(EXP / 'replay.json', dict(at=stamp(), protocol_sha256=sha(PLAN), cases=rows, GPU_hours=0))


if __name__ == '__main__':
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=('isolate', 'render'))
    args = parser.parse_args()
    (isolate if args.step == 'isolate' else render)(read(PLAN)['photos'])

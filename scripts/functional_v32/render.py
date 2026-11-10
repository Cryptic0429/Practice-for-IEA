"""Isolate held sources and render anonymized contrast comparisons."""
import argparse
import hashlib
import shutil
from itertools import combinations
from pathlib import Path

from PIL import Image

from functional_v1.common import ROOT, read, sha, stamp, write
from functional_v15.editor import load_rgb, save
from functional_v24.probe import SourceMatcher, protected_sources
from functional_v30.render import panel
from functional_v32.editor import RECIPES, apply

NAME = 'functional-v32-20261107'
EXP = ROOT / 'experiments' / NAME
DATA = ROOT / 'data/training' / NAME
PLAN = EXP / 'protocol.json'


def isolate(photos):
    protected = protected_sources()
    for version, name in [(24, 'reviewed-candidates'), (26, 'reviewed-candidates'), (27, 'held-cases')]:
        for row in read(ROOT / f'data/training/functional-v{version}-20261107/{name}.json'):
            protected[str(ROOT / row['image'])] = f'r{version}'
    for version in (29, 30, 31):
        for row in read(ROOT / f'experiments/functional-v{version}-20261107/replay.json')['cases']:
            protected[str(ROOT / row['source'])] = f'r{version}'
            protected[str(ROOT / row.get('path', row['source']))] = f'r{version}'
    for row in photos:
        if row['cohort'] == 'fit':
            protected[str(ROOT / row['path'])] = 'fit'
    missing = {p: kind for p, kind in protected.items() if not Path(p).is_file()}
    if set(missing.values()) - {'GIER_val', 'GIER_test'}:
        raise ValueError('Missing protected internal original')
    available = [p for p in protected if p not in missing]
    held = [p for p in photos if p['cohort'] == 'held']
    matcher = SourceMatcher()
    flags = []
    for row in held:
        for other in available:
            match = matcher.compare(ROOT / row['path'], other)
            if match:
                flags.append(dict(id=row['id'], other=other, match=match))
        print('Compared', row['id'], flush=True)
    for a, b in combinations(held, 2):
        match = matcher.compare(ROOT / a['path'], ROOT / b['path'])
        if match or a['group_id'] == b['group_id']:
            flags.append(dict(id=a['id'], other=b['id'], match=match))
    write(EXP / 'isolation.json', dict(at=stamp(), protocol_sha256=sha(PLAN),
          available_protected=len(available), missing=missing, flags=flags, comparisons=len(matcher.pairs),
          source_hashes={r['id']: sha(ROOT / r['path']) for r in held}, GPU_hours=0))


def render(photos):
    isolation = read(EXP / 'isolation.json')
    if isolation['flags'] or isolation['protocol_sha256'] != sha(PLAN):
        raise ValueError('Source isolation has not passed')
    rows = []
    for photo in photos:
        original = ROOT / photo['path']
        digest = sha(original)
        if photo['cohort'] == 'held':
            if digest != isolation['source_hashes'][photo['id']]:
                raise ValueError('Held original changed')
            copy = DATA / 'held-inputs' / (photo['id'] + original.suffix)
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, copy)
            original = copy
        pixels = load_rgb(original)
        for direction, requests in photo['requests'].items():
            case = f"{photo['id']}-{direction}"
            order = sorted(RECIPES, key=lambda name: hashlib.sha256(f'{case}/{name}'.encode()).digest())
            images = {}
            for name in order:
                edited = apply(pixels, direction, name)
                path = EXP / 'images' / f'{case}-{name}.png'
                save(path, edited)
                images[name] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path))
            board, box = panel(Image.fromarray(pixels), [Image.open(ROOT / v['path']) for v in images.values()], photo['crop'])
            board_path = EXP / 'boards' / f'{case}.png'
            board_path.parent.mkdir(exist_ok=True)
            board.save(board_path)
            rows.append(dict(id=case, photo_id=photo['id'], group_id=photo['group_id'], cohort=photo['cohort'],
                             domain=photo['domain'], direction=direction, requests=requests, candidate_order=order,
                             source=str(original.relative_to(ROOT)), source_sha256=digest, images=images,
                             native_crop=box, board=str(board_path.relative_to(ROOT)), board_sha256=sha(board_path)))
        print('Rendered', photo['id'], flush=True)
    write(EXP / 'replay.json', dict(at=stamp(), protocol_sha256=sha(PLAN), cases=rows, GPU_hours=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=('isolate', 'render'))
    args = parser.parse_args()
    (isolate if args.step == 'isolate' else render)(read(PLAN)['photos'])

"""Inspect stored GIER training originals and replay existing edits on CPU."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from functional_v1.common import ROOT, read, sha, stamp, write
from functional_v6.isolation import SourceMatcher
from functional_v15.editor import load_rgb
from functional_v15.outcomes import execute
from functional_v21.texture import features

EXP = ROOT / 'experiments/functional-v24-20261107'
GIER = ROOT / 'repo/datasets/GIER'
GROUPS = ROOT / 'data/stage1/splits/image-edit-3k-v1/group-assignments.json'
PROTOCOL = ROOT / 'experiments/functional-v22-20261107/training-protocol.json'
CUES = {
    'portrait': r'\b(portrait|face|headshot|selfie|skin|myself|photo of me)\b',
    'landscape': r'\b(landscape|sky|foreground|mountains?|sunset|sunrise|scenery)\b',
}


def bound_path(spec):
    path = ROOT / spec['path']
    if sha(path) != spec['sha256']:
        raise ValueError(f'Frozen source changed: {path}')
    return path


def draw_cell(sheet, image, x, y, size, title):
    ImageDraw.Draw(sheet).text((x + 5, y + 4), title, fill='black')
    tile = ImageOps.contain(image, (size[0] - 10, size[1] - 22))
    sheet.paste(tile, (x + (size[0] - tile.width) // 2, y + 20 + (size[1] - 22 - tile.height) // 2))


def scan():
    protocol = read(PROTOCOL)
    prior = read(bound_path(protocol['sources']['train']))
    old_groups = {c['group_id'] for c in prior}
    old_names = {Path(c.get('canonical_photo') or c['image']).name for c in prior}
    annotations = read(GIER / 'GIER.json')
    split = read(GIER / 'split.json')
    by_name = defaultdict(list)
    for index in split['train']:
        by_name[Path(annotations[index]['input']).name].append(index)
    reserved = {Path(annotations[i]['input']).name for k in ('test', 'val') for i in split[k]}
    rows = []
    for group in read(GROUPS):
        if group['partition'] != 'training_pool' or group['group_id'] in old_groups:
            continue
        for name in group['source_paths']:
            path = Path(name)
            if path.name not in by_name or path.name in reserved or path.name in old_names:
                continue
            with Image.open(path) as source:
                exif = source.getexif()
                sub = exif.get_ifd(34665)
                thumbnail = ImageOps.exif_transpose(source).convert('RGB')
                thumbnail.thumbnail((256, 256))
                lum = np.asarray(thumbnail) @ np.array([.2126, .7152, .0722])
                indices = by_name[path.name]
                hints = ' '.join(annotations[i]['request'] or '' for i in indices)
                rows.append(dict(
                    id=f'p{len(rows) + 1:03d}', source=str(path.relative_to(ROOT)),
                    group_id=group['group_id'], annotation_indices=indices,
                    size=list(source.size), iso=sub.get(34855) or exif.get(34855),
                    exif_camera_model_present=bool(exif.get(272)),
                    cues=[d for d, pattern in CUES.items() if re.search(pattern, hints, re.I)],
                    median_luma=round(float(np.median(lum)), 2),
                    dark_share=round(float(np.mean(lum < 38)), 5),
                    dark_chroma_residual=round(float(features(path)[6]), 6),
                ))
    short = []
    seen = set()
    for cue in ('portrait', 'landscape', 'uncued'):
        eligible = [r for r in rows if (cue in r['cues'] if cue != 'uncued' else not r['cues'])
                    and r['dark_share'] >= .13 and r['median_luma'] <= 128]
        for reverse in (True, False):
            take = []
            for row in sorted(eligible, key=lambda r: r['dark_chroma_residual'], reverse=reverse):
                if row['group_id'] not in seen:
                    seen.add(row['group_id'])
                    take.append(row)
                    if len(take) == 8:
                        break
            short.extend(take)
    report = dict(at=stamp(), sources={str(p.relative_to(ROOT)): sha(p) for p in (
        GROUPS, GIER / 'GIER.json', GIER / 'split.json', PROTOCOL)},
        row_count=len(rows), iso_available=sum(bool(r['iso']) for r in rows),
        camera_model_available=sum(r['exif_camera_model_present'] for r in rows),
        shortlist=[r['id'] for r in short], rows=rows,
        rule='Stage1 training_pool and official GIER train. Exact prior group/name and official reserved name exclusions. Brightness, residual and text cues used only for triage; visual domain and quality remain unverified.')
    write(EXP / 'scan.json', report)
    for part in range(0, len(short), 16):
        sheet = Image.new('RGB', (960, 752), 'white')
        for i, row in enumerate(short[part:part+16]):
            with Image.open(ROOT / row['source']) as image:
                draw_cell(sheet, ImageOps.exif_transpose(image).convert('RGB'), i % 4 * 240, i // 4 * 188,
                          (240, 188), f"{row['id']} | dark {row['dark_share']:.2f} | residual {row['dark_chroma_residual']:.4f}")
        sheet.save(EXP / f'shortlist-{part//16+1}.jpg', quality=91)
    print('Scanned', len(rows), 'shortlisted', len(short))


def protected_sources():
    protocol = read(PROTOCOL)
    protected = {}
    for kind in ('sources', 'sealed_acceptance_sources'):
        for name, spec in protocol[kind].items():
            for row in read(bound_path(spec)):
                image = row['image']
                if name == 'train':
                    image = row.get('canonical_photo') or image
                protected[str(ROOT / image)] = name
    for row in read(GROUPS):
        if row['partition'] != 'training_pool':
            for path in row['source_paths']:
                protected.setdefault(path, row['partition'])
    annotations = read(GIER / 'GIER.json')
    split = read(GIER / 'split.json')
    for key in ('val', 'test'):
        for i in split[key]:
            path = GIER / 'images' / annotations[i]['input']
            protected.setdefault(str(path), f'GIER_{key}')
    return protected


def isolate(selected):
    protected = protected_sources()
    missing = {path for path in protected if not Path(path).is_file()}
    matcher = SourceMatcher()
    flags = []
    for row in selected:
        source = ROOT / row['source']
        for path, kind in protected.items():
            if path in missing:
                continue
            match = matcher.compare(source, path)
            if match:
                flags.append(dict(source=row['id'], protected=path, split=kind, match=match))
        print('Isolated', row['id'], 'against', len(protected)-len(missing), 'paths', flush=True)
    for a, b in combinations(selected, 2):
        match = matcher.compare(ROOT / a['source'], ROOT / b['source'])
        if match or a['group_id'] == b['group_id']:
            flags.append(dict(source=a['id'], other_candidate=b['id'], match=match))
    write(EXP / 'isolation.json', dict(at=stamp(), plan_sha256=sha(EXP/'plan.json'),
          protected_paths_by_split=dict(Counter(protected.values())), missing=sorted(missing), flags=flags,
          candidate_source_hashes={r['id']: sha(ROOT/r['source']) for r in selected},
          comparisons=len(matcher.pairs),
          method='Inherited pHash/gray correlation and ORB geometry; protected inputs only. Can miss other crops or frames from the same shoot.'))


def render(selected, plan_path, output_dir):
    candidate_dir = output_dir / 'candidates'
    candidate_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for row in selected:
        source = ROOT / row['source']
        rgb = load_rgb(source)
        height, width = rgb.shape[:2]
        cx, cy = row['crop_center_fraction']
        cw, ch = min(290, width), min(190, height)
        left = max(0, min(width - cw, round(cx * width - cw / 2)))
        top = max(0, min(height - ch, round(cy * height - ch / 2)))
        box = (left, top, left + cw, top + ch)
        full = Image.new('RGB', (1200, 610), 'white')
        draw = ImageDraw.Draw(full)
        draw.text((8, 8), row['id'] + ' | ' + row['instructions']['en'], fill='black')
        choices = [dict(action='keep')] + [dict(action='edit', parameters=p) for p in row['edits']]
        stored = []
        for i, decision in enumerate(choices):
            result = execute(rgb, decision)
            image = Image.fromarray(result['result'])
            output = candidate_dir / f"{row['id']}-{i}.png"
            image.save(output)
            label = 'Input' if not i else f'Candidate {i}'
            draw_cell(full, image, i * 300, 30, (300, 300), label)
            draw.text((i * 300 + 5, 345), 'Native crop | 1 output pixel per image pixel', fill='black')
            full.paste(image.crop(box), (i * 300 + 5, 366))
            stored.append(dict(index=i, path=str(output.relative_to(ROOT)), sha256=sha(output), decision=decision))
        panel = output_dir / f"{row['id']}.png"
        full.save(panel)
        artifacts.append(dict(id=row['id'], source_sha256=sha(source), native_crop_box=box,
                              full_frame_size=[width, height], candidates=stored,
                              editor=result['editor'], panel=panel.name, panel_sha256=sha(panel)))
    write(output_dir/'replay.json', dict(at=stamp(), plan_sha256=sha(plan_path), cases=artifacts, GPU_hours=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('scan', 'isolate', 'render'))
    parser.add_argument('--refine', action='store_true', help='Render the separately logged follow-up comparisons')
    args = parser.parse_args()
    if args.refine and args.stage != 'render':
        parser.error('--refine requires render')
    if args.stage == 'scan':
        scan()
    else:
        plan_path = EXP / ('refinement-plan.json' if args.refine else 'plan.json')
        plan = read(plan_path)
        scan_file = EXP/'scan.json'
        if sha(scan_file) != plan['scan_sha256']:
            raise ValueError('Scan has changed since plan creation')
        rows = {row['id']: row for row in read(scan_file)['rows']}
        selected = [dict(rows[c['id']], **{k: v for k, v in c.items() if k != 'id'}) for c in plan['cases']]
        if args.stage == 'isolate':
            isolate(selected)
        else:
            render(selected, plan_path, EXP/'refinements' if args.refine else EXP)


if __name__ == '__main__':
    main()

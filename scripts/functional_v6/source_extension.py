"""Immutable additional source retrieval. No targets, model output or label admission."""
import argparse
import hashlib
import json
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from functional_v1.common import ROOT, read, write, append, sha, stamp
from functional_v1.curate import font
from .source_bank import EXP, DATA, ALLOW, EXCLUDE, family, norm


def prepare(split):
    out = EXP/'source-extension'/split
    assert not (out/'manifest.json').exists(), 'Preserve each source retrieval snapshot'
    initial = read(EXP/'source-review/manifest.json')
    excluded = {r['group_id'] for r in initial['candidates']}
    for directory, names in [('functional-v1-20261104', ('development-cases.json', 'acceptance-cases.json')),
                              ('functional-v2-20261104', ('development-final-cases.json', 'acceptance-final-cases.json'))]:
        for name in names:
            excluded.update(c['group_id'] for c in read(ROOT/'data/training'/directory/name))
    excluded.update(c['group_id'] for c in read(ROOT/'data/training/functional-v4-20261105/cases.json'))
    assignments = ROOT/'data/stage1/splits/image-edit-3k-v1/group-assignments.json'
    groups = {p: g for g in read(assignments) for p in g['source_paths']}
    original = ROOT/'data/sources/gier/GIER.json'
    annotated = defaultdict(list)
    for i, m in enumerate(read(original)):
        fields = [('request', None, m.get('request', '')), ('detailed_request', None, m.get('detailed_request', ''))]
        for f in ('expert_summary', 'amateur_summary'):
            fields.extend((f, j, t) for j, t in enumerate(m.get(f, [])))
        annotated[m['input']].extend(dict(metadata_index=i, field=f, item_index=j, original_text=t)
                                    for f, j, t in fields if isinstance(t, str) and t.strip())
    ready = ROOT/'data/stage1/ready/GIER-Edit'/('sft_train.json' if split == 'train' else 'sft_test.json')
    partition = 'training_pool' if split == 'train' else 'test_reserved'
    buckets = defaultdict(list)
    unmatched = 0
    for row in read(ready):
        extra = row['extra_info']; text = extra['instruction']; path = extra['origin_image_path']
        if not text.strip() or not ALLOW.search(text) or EXCLUDE.search(text): continue
        group = groups[path]
        if group['partition'] != partition or group['group_id'] in excluded: continue
        matches = [a for a in annotated[Path(path).name] if norm(a['original_text']) == norm(text)]
        if not matches:
            unmatched += 1
            continue
        buckets[family(text)].append(dict(record_id=row['id'], group_id=group['group_id'],
            source_image=path, instruction=text, instruction_type=extra['inst_type'], split=split,
            source_partition=partition, screened_family=family(text), native_annotation=matches,
            original_reference=extra.get('reference_image_path'), status='source_only_no_label_or_model_output'))
    for bucket in buckets.values():
        bucket.sort(key=lambda r: hashlib.sha256(('functional-v6-source-extension'+r['record_id']).encode()).hexdigest())
    chosen = []; seen = set(excluded); positions = Counter()
    while True:
        before = len(chosen)
        for f in ('brown_tone', 'white_balance', 'contrast', 'saturation', 'lighting'):
            bucket = buckets[f]
            while positions[f] < len(bucket):
                r = bucket[positions[f]]; positions[f] += 1
                if r['group_id'] in seen: continue
                seen.add(r['group_id'])
                chosen.append(dict(r, id=f'{split}-extension-{len(chosen)+1:03d}', source_sha256=sha(r['source_image'])))
                break
        if len(chosen) == before: break
    sheets = []
    for offset in range(0, len(chosen), 80):
        subset = chosen[offset:offset+80]
        canvas = Image.new('RGB', (1920, 2200), 'white'); d = ImageDraw.Draw(canvas)
        d.text((12, 8), split+' SOURCE OVERVIEW: domain triage only; not quality/label review', font=font(25), fill='black')
        for i, r in enumerate(subset):
            x, y = i%8*240, 50+i//8*215
            d.text((x+4, y), r['id'], font=font(16), fill='black')
            im = ImageOps.contain(ImageOps.exif_transpose(Image.open(r['source_image'])).convert('RGB'), (230, 175))
            canvas.paste(im, (x+5+(230-im.width)//2, y+27+(175-im.height)//2))
        path = out/f'overview-{offset//80+1:02d}.jpg'; path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, quality=96, subsampling=0)
        sheets.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path), source_ids=[r['id'] for r in subset], purpose='domain_triage_only'))
    write(out/'manifest.json', dict(at=stamp(), split=split, candidates=chosen, sheets=sheets,
        source_files={str(p.relative_to(ROOT)): sha(p) for p in (ready, assignments, original)},
        initial_source_manifest_sha256=sha(EXP/'source-review/manifest.json'),
        unmatched_annotation_rows_excluded=unmatched,
        scope='Original stored English unchanged; source matching only. No reference viewed, no model inference, no parameters or training labels admitted. Overview pictures only support domain triage; selected sources need request/detail review, isolation and executed-target visual admission.',
        final_acceptance_outputs_closed=True))
    print(split, 'source candidates', len(chosen))


def detail(split, ids):
    out = EXP/'source-extension'/split
    manifest = read(out/'manifest.json'); bank = {r['id']: r for r in manifest['candidates']}
    selected = [bank[f'{split}-extension-{int(i):03d}'] for i in ids.split(',')]
    key = hashlib.sha256(','.join(r['id'] for r in selected).encode()).hexdigest()[:10]
    destination = out/f'detail-{key}.json'
    assert not destination.exists(), 'Do not overwrite source observation sheets'
    sheets = []
    for offset in range(0, len(selected), 12):
        subset = selected[offset:offset+12]
        canvas = Image.new('RGB', (1800, 1650), 'white'); d = ImageDraw.Draw(canvas)
        d.text((12, 8), split+' SOURCE/REQUEST: no label approved', font=font(26), fill='black')
        for i, r in enumerate(subset):
            x, y = i%4*450, 55+i//4*525
            d.text((x+8, y), r['id'], font=font(22), fill='black')
            for j, line in enumerate(textwrap.wrap(r['instruction'], 42)[:3]):
                d.text((x+8, y+28+j*22), line, font=font(18), fill='black')
            im = ImageOps.contain(ImageOps.exif_transpose(Image.open(r['source_image'])).convert('RGB'), (430, 385))
            canvas.paste(im, (x+10+(430-im.width)//2, y+110+(385-im.height)//2))
        path = out/f'detail-{key}-{offset//12+1:02d}.jpg'
        canvas.save(path, quality=96, subsampling=0)
        sheets.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path), source_ids=[r['id'] for r in subset]))
    write(destination, dict(at=stamp(), manifest_sha256=sha(out/'manifest.json'), sheets=sheets))
    print(destination)


def record(split, notes_path, detail_manifest):
    out = EXP/'source-extension'/split
    manifest = read(out/'manifest.json'); bank = {r['id']: r for r in manifest['candidates']}
    detail_info = read(detail_manifest)
    assert detail_info['manifest_sha256'] == sha(out/'manifest.json')
    sheets = {sid: s for s in detail_info['sheets'] for sid in s['source_ids']}
    dest = out/'observations.jsonl'
    done = {r['id'] for r in map(json.loads, dest.read_text().splitlines())} if dest.exists() else set()
    for item in read(notes_path):
        assert item['id'] not in done and len(item['evidence']) >= 15
        assert item['domain'] in ('portrait', 'landscape', 'reject')
        assert item['request_scope'] in ('global', 'local', 'ambiguous', 'unsupported')
        assert isinstance(item['source_usable'], bool)
        c, s = bank[item['id']], sheets[item['id']]
        assert sha(c['source_image']) == c['source_sha256'] and sha(ROOT/s['path']) == s['sha256']
        append(dest, dict(item, group_id=c['group_id'], source_sha256=c['source_sha256'],
            sheet=s['path'], sheet_sha256=s['sha256'], observed_at=stamp(),
            reviewer='Codex direct source/request classification, AI assisted; not target or functional-quality approval'))
        done.add(item['id'])
    print('Source/request observations recorded, no labels admitted')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['prepare', 'detail', 'record'])
    ap.add_argument('--split', choices=['train', 'acceptance'], required=True)
    ap.add_argument('--ids'); ap.add_argument('--notes', type=Path); ap.add_argument('--detail-manifest', type=Path)
    args = ap.parse_args()
    if args.command == 'prepare': prepare(args.split)
    elif args.command == 'detail': detail(args.split, args.ids)
    else: record(args.split, args.notes, args.detail_manifest)

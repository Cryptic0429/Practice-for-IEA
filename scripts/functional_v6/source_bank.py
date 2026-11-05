"""Retrieve original annotated global requests from the full eligible pool, not just old3k."""
import argparse
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from functional_v1.common import ROOT, read, write, append, sha, stamp
from functional_v1.curate import font

NAME = 'functional-v6-20261105'
EXP = ROOT/'experiments'/NAME
DATA = ROOT/'data/training'/NAME
ALLOW = re.compile(r'bright|lighten|darken|shadow|highlight|expos|satur|vivid|vibran|colou?rful|contrast|warmer|cooler|warm tone|cool tone|brown|sepia', re.I)
EXCLUDE = re.compile(r'remov|eras|crop|sharp|blurr|noise|smooth|background|black and white|black.white|blemish|retouch|red.eye|teeth|resize|rotate|text|watermark|face|skin|object|replace|eyes|photoshop|filter|hue', re.I)


def family(text):
    if re.search(r'sepia|brown tone', text, re.I): return 'brown_tone'
    if re.search(r'warmer|cooler|warm tone|cool tone|color warmth', text, re.I): return 'white_balance'
    if re.search(r'contrast', text, re.I): return 'contrast'
    if re.search(r'satur|vivid|vibran|colou?rful|enhance.*colou?r', text, re.I): return 'saturation'
    return 'lighting'


def norm(text):
    return ' '.join(text.split()).lower()


def prepare(extend_development=False):
    out = EXP/'source-review'
    previous = read(out/'manifest.json') if extend_development else None
    if not extend_development and (out/'manifest.json').exists():
        raise RuntimeError('Preserve source retrieval; use a recorded extension if needed')
    if extend_development:
        assert not (DATA/'source-splits.json').exists()
        write(out/f'manifest-before-dev-extension-{len(previous["candidates"])}.json', previous)
    ready = ROOT/'data/stage1/ready/GIER-Edit/sft_train.json'
    assignments = ROOT/'data/stage1/splits/image-edit-3k-v1/group-assignments.json'
    original = ROOT/'data/sources/gier/GIER.json'
    groups = {p: g for g in read(assignments) for p in g['source_paths']}
    annotated = defaultdict(list)
    for i, m in enumerate(read(original)):
        fields = [('request', None, m.get('request', '')), ('detailed_request', None, m.get('detailed_request', ''))]
        for field in ('expert_summary', 'amateur_summary'):
            fields.extend((field, j, text) for j, text in enumerate(m.get(field, [])))
        annotated[m['input']].extend(dict(metadata_index=i, field=f, item_index=j, original_text=t)
                                     for f, j, t in fields if isinstance(t, str) and t.strip())
    prior_holdout = set()
    for directory, splits in [('functional-v1-20261104', ('development-cases.json', 'acceptance-cases.json')),
                              ('functional-v2-20261104', ('development-final-cases.json', 'acceptance-final-cases.json'))]:
        for split in splits:
            # Existing developmental photos may receive new native requests, with history disclosed.
            # Acceptance photos remain excluded. This does not create an independent pristine test.
            if extend_development and split.startswith('development'):
                continue
            prior_holdout.update(c['group_id'] for c in read(ROOT/'data/training'/directory/split))
    prior_course = {c['group_id'] for c in read(ROOT/'data/training/functional-v4-20261105/cases.json')}
    candidates = defaultdict(list)
    unverified = 0
    for row in read(ready):
        extra = row['extra_info']
        text = extra['instruction']
        if not text.strip() or not ALLOW.search(text) or EXCLUDE.search(text): continue
        path = extra['origin_image_path']
        group = groups[path]
        part = group['partition']
        if part not in ('training_pool', 'validation_reserved') or group['group_id'] in prior_holdout: continue
        split = 'train' if part == 'training_pool' else 'development'
        if split == 'train' and group['group_id'] in prior_course: continue
        matches = [x for x in annotated[Path(path).name] if norm(x['original_text']) == norm(text)]
        if not matches:
            unverified += 1
            continue
        candidates[split].append(dict(record_id=row['id'], group_id=group['group_id'], source_image=path,
            instruction=text, instruction_type=extra['inst_type'], screened_family=family(text), split=split,
            native_annotation=matches, original_reference=extra.get('reference_image_path'),
            status='source_candidate_only_no_target_or_training_admission'))
    chosen = []
    limits = [('development', 999)] if extend_development else [('train', 240), ('development', 80)]
    old_groups = {r['group_id'] for r in previous['candidates']} if previous else set()
    for split, maximum in limits:
        buckets = defaultdict(list)
        for row in candidates[split]: buckets[row['screened_family']].append(row)
        for bucket in buckets.values(): bucket.sort(key=lambda r: hashlib.sha256((NAME+r['record_id']).encode()).hexdigest())
        seen = set(old_groups); selected = []; positions = Counter()
        old_count = sum(r['split'] == split for r in previous['candidates']) if previous else 0
        while len(selected) < maximum:
            before_count = len(selected)
            for f in ('brown_tone', 'white_balance', 'contrast', 'saturation', 'lighting'):
                bucket = buckets[f]
                while positions[f] < len(bucket):
                    row = bucket[positions[f]]; positions[f] += 1
                    if row['group_id'] in seen: continue
                    seen.add(row['group_id'])
                    row = dict(row, id=f'{split}-native-{old_count+len(selected)+1:03d}', source_sha256=sha(row['source_image']))
                    selected.append(row)
                    break
                if len(selected) == maximum: break
            if len(selected) == before_count: break
        chosen.extend(selected)
    assert not ({r['group_id'] for r in chosen if r['split'] == 'train'} & {r['group_id'] for r in chosen if r['split'] == 'development'})
    sheets = []
    for split in ('train', 'development'):
        rows = [r for r in chosen if r['split'] == split]
        for offset in range(0, len(rows), 12):
            subset = rows[offset:offset+12]
            canvas = Image.new('RGB', (1800, 1650), 'white'); d = ImageDraw.Draw(canvas)
            d.text((12, 10), split+' | source/request classification only; no editing labels', font=font(26), fill='black')
            for i, r in enumerate(subset):
                x, y = i%4*450, 60+i//4*525
                d.text((x+8, y), r['id'], font=font(22), fill='black')
                lines = __import__('textwrap').wrap(r['instruction'], 41)
                for j, line in enumerate(lines[:3]): d.text((x+8, y+29+j*22), line, font=font(18), fill='black')
                image = ImageOps.contain(ImageOps.exif_transpose(Image.open(r['source_image'])).convert('RGB'), (430, 385))
                canvas.paste(image, (x+10+(430-image.width)//2, y+110+(385-image.height)//2))
            file = out/f'{"extra-" if extend_development else ""}{split}-{offset//12+1:02d}.jpg'
            file.parent.mkdir(parents=True, exist_ok=True); canvas.save(file, quality=96, subsampling=0)
            sheets.append(dict(path=str(file.relative_to(ROOT)), sha256=sha(file), source_ids=[r['id'] for r in subset]))
    additions = {s: sum(r['split'] == s for r in chosen) for s in ('train', 'development')}
    if previous:
        chosen = previous['candidates']+chosen
        sheets = previous['sheets']+sheets
    write(out/'manifest.json', dict(created_at=stamp(), candidates=chosen, sheets=sheets,
         source_files={str(p.relative_to(ROOT)): sha(p) for p in (ready, assignments, original)},
         retrieved_counts={s: dict(Counter(r['screened_family'] for r in chosen if r['split'] == s)) for s in ('train', 'development')},
         unmatched_native_annotation_rows_excluded=unverified,
         policy='Original stored English is not rewritten. Match to GIER request/expert/amateur annotations verified after whitespace normalization. Keyword retrieval does not establish global edit scope or photographic domain; both require direct observation. No new data used in training yet. Selected sources still need near-duplicate isolation before freeze, targets and bilingual requests then need direct visual admission. Historical original validation pool, not newly collected independent user photographs.',
         prospective_balance='Aim80 new native training photos40portrait/40landscape,20native development photos10/10; real admitted counts and task coverage govern. Never fill shortages with unreviewed labels. No new acceptance source opened.'))
    if previous:
        manifest = read(out/'manifest.json')
        manifest['extension'] = dict(at=stamp(), added=additions,
            reason='Initial80 developer sources contain very few landscape photographs. Exhaust remaining original validation-reserved native-request sources, including historically used development photos. Existing observed candidates/sheets kept unchanged. No acceptance group eligible; no output inference or training.',
            previous_manifest_sha256=sha(out/f'manifest-before-dev-extension-{len(previous["candidates"])}.json'))
        write(out/'manifest.json', manifest)
    print(additions)


def record(notes):
    manifest = read(EXP/'source-review/manifest.json')
    byid = {r['id']: r for r in manifest['candidates']}
    sheet = {cid: s for s in manifest['sheets'] for cid in s['source_ids']}
    dest = EXP/'source-review/observations.jsonl'
    done = {r['id'] for r in map(__import__('json').loads, dest.read_text().splitlines())} if dest.exists() else set()
    for item in read(notes):
        assert item['id'] not in done
        assert item['domain'] in ('portrait', 'landscape', 'reject')
        assert item['request_scope'] in ('global', 'local', 'ambiguous', 'unsupported')
        assert isinstance(item['source_usable'], bool) and len(item['evidence']) >= 15
        c, s = byid[item['id']], sheet[item['id']]
        assert sha(ROOT/s['path']) == s['sha256'] and sha(c['source_image']) == c['source_sha256']
        append(dest, dict(item, group_id=c['group_id'], source_sha256=c['source_sha256'],
                          sheet=s['path'], sheet_sha256=s['sha256'], observed_at=stamp(),
                          reviewer='Codex direct source/request visual classification; AI assisted, not a label-quality approval'))
        done.add(item['id'])
    print('source observations recorded; no editing labels admitted')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['prepare', 'extend-development', 'record'], default='prepare', nargs='?')
    ap.add_argument('--notes', type=Path)
    args = ap.parse_args()
    if args.command == 'record': record(args.notes)
    else: prepare(args.command == 'extend-development')

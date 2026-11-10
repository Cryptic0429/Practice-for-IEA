"""Preserve direct repair observations and exact native review assets."""
from pathlib import Path
import json, hashlib, datetime, argparse
from .native_labels import ROOT, EXP, CRITICAL_REGIONS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(bundle):
    from PIL import Image, ImageDraw
    source = EXP / 'label-repairs' / bundle / 'manifest.json'
    dest = EXP / 'native-review/repairs' / bundle
    assert not (dest / 'manifest.json').exists()
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in json.loads(source.read_text())['items']:
        images = []
        for role, path, expected in [('INPUT', item['input']['path'], item['input']['sha256'])] + [
                (r['candidate'], r['image'], r['image_sha256']) for r in item['results']]:
            assert digest(ROOT / path) == expected
            with Image.open(ROOT / path) as im:
                rgb = im.convert('RGB'); w, h = rgb.size
                regions = [('whole', [0, 0, w, h])] if max(w, h) <= 2000 and w * h <= 3500000 else CRITICAL_REGIONS[item['id']]
                for region, box in regions:
                    cropped = rgb.crop(box)
                    target = dest / f'{item["id"]}-{role}-{region}.png'
                    assert not target.exists()
                    cropped.save(target, compress_level=1)
                    with Image.open(target) as check:
                        assert check.size == cropped.size and check.convert('RGB').tobytes() == cropped.tobytes()
                    images.append(dict(role=role, region=region, path=str(target.relative_to(ROOT)),
                                       sha256=digest(target), size=list(cropped.size), box=box,
                                       source=path, source_sha256=expected, source_size=[w, h]))
        rows.append(dict(id=item['id'], images=images))
    (dest / 'manifest.json').write_text(json.dumps(dict(items=rows, source_manifest_sha256=digest(source),
        scope='Exact native copies or explicit listed crops. No resizing/adjustment. No admission.'), indent=2) + '\n')
    print('Exported native repair assets', bundle, len(rows))


def record(bundle, cid, candidate, stage, completion, quality, evidence):
    assert stage in ('overview', 'native') and completion in 'CPFU' and quality in 'AmMU' and len(evidence) > 30
    manifest = EXP / 'label-repairs' / bundle / 'manifest.json'
    item = next(i for i in json.loads(manifest.read_text())['items'] if i['id'] == cid)
    result = next(r for r in item['results'] if r['candidate'] == candidate)
    assert digest(ROOT / item['sheet']) == item['sheet_sha256']
    assert digest(ROOT / result['image']) == result['image_sha256']
    assert digest(ROOT / item['input']['path']) == item['input']['sha256']
    regions = None
    if stage == 'native':
        native_manifest = EXP / 'native-review/repairs' / bundle / 'manifest.json'
        native = json.loads(native_manifest.read_text())
        assert native['source_manifest_sha256'] == digest(manifest)
        regions = [i for i in next(i for i in native['items'] if i['id'] == cid)['images'] if i['role'] in ('INPUT', candidate)]
        assert regions and all(digest(ROOT / i['path']) == i['sha256'] for i in regions)
    dest = EXP / 'repair-observations.jsonl'
    old = dest.read_text() if dest.exists() else ''
    key = (bundle, cid, candidate, stage)
    assert key not in {(i['bundle'], i['id'], i['candidate'], i['stage']) for i in map(json.loads, old.splitlines())}
    row = dict(bundle=bundle, id=cid, candidate=candidate, stage=stage, completion=completion,
        quality=quality, evidence=evidence, at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        manifest_sha256=digest(manifest), input=item['input'], result=result, regions=regions,
        reviewer='Codex direct AI review, not human/user approval', label_admitted=False)
    temp = dest.with_suffix('.tmp')
    temp.write_text(old + json.dumps(row, ensure_ascii=False) + '\n'); temp.replace(dest)
    print('Repair observation preserved', cid, candidate, stage, completion, quality)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--bundle', required=True)
    prepare(parser.parse_args().bundle)

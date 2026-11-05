"""Run on SJTU1: validate source assets and prepare traceable, safe SFT copies."""
import collections
import concurrent.futures
import copy
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image

ROOT = Path('/mnt/sdc/zhc/IEA')
REPO = ROOT/'repo'
OUT = ROOT/'data/stage1'
TOOLS = set('exposure brightness contrast natural_contrast highlights shadows whites blacks saturation vibrance temperature tint sharpness vignette fade grain'.split())
_IMAGE_CACHE = {}
_REPAIRS = {}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def resolve(value):
    value = value.replace('\\','/')
    suffix = value[value.index('datasets/'):]
    path = (REPO/suffix).resolve()
    if not path.is_relative_to(REPO/'datasets'):
        raise ValueError('Path escapes datasets')
    replacement = _REPAIRS.get(str(path), {}).get('replacement')
    if replacement:
        assert Path(replacement).resolve().is_relative_to(OUT/'compatible-images')
        return replacement
    return str(path)


def image_info(path):
    try:
        content = Path(path).read_bytes()
        content_sha = hashlib.sha256(content).hexdigest()
        cached = _IMAGE_CACHE.get(path, {})
        if cached.get('sha256') == content_sha and 'pixel_sha256' in cached:
            return path, cached
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            rgb = im.convert('RGB')
            h = hashlib.sha256()
            h.update(str(rgb.size).encode())
            h.update(rgb.tobytes())
            return path, dict(bytes=len(content), sha256=content_sha,
                              pixel_sha256=h.hexdigest(), width=rgb.width, height=rgb.height)
    except Exception as e:
        return path, dict(error=f'{type(e).__name__}: {str(e)}')


def main():
    global _IMAGE_CACHE, _REPAIRS
    if (OUT/'image-manifest.json').exists():
        _IMAGE_CACHE = json.loads((OUT/'image-manifest.json').read_text())
    if (OUT/'image-repairs.json').exists():
        _REPAIRS = json.loads((OUT/'image-repairs.json').read_text())
    datasets, source_stats, paths, errors = {}, {}, set(), []
    for task in ['GIER-Edit','GIER-Summary']:
        for split in ['train','test']:
            source = REPO/'datasets'/task/f'sft_{split}.json'
            raw = json.loads(source.read_text())
            key = f'{task}/{split}'
            ids = collections.Counter(r['id'] for r in raw)
            source_stats[key] = dict(rows=len(raw), unique_source_ids=len(ids),
                                    repeated_id_rows=sum(c-1 for c in ids.values()),
                                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            normalized=[]
            for idx, item in enumerate(raw):
                row = copy.deepcopy(item)
                row['source_id'] = row['id']
                row['id'] = f'{task}:{split}:{idx:06d}'
                row['source_row'] = idx
                row['images'] = [resolve(p) for p in row['images']]
                for k in ['origin_image_path','reference_image_path']:
                    row['extra_info'][k] = resolve(row['extra_info'][k])
                paths.update(row['images'])
                paths.update(row['extra_info'][k] for k in ['origin_image_path','reference_image_path'])
                try:
                    conv = row['conversations']
                    assert [c['role'] for c in conv] == ['system','user','assistant']
                    assert all(isinstance(c['content'],str) and c['content'].strip() for c in conv)
                    assert sum(c['content'].count('<image>') for c in conv) == len(row['images'])
                    assert '<answer>' in conv[-1]['content'] and '</answer>' in conv[-1]['content']
                    if task=='GIER-Edit':
                        answer = re.search(r'<answer>\s*```json\s*(.*?)\s*```\s*</answer>', conv[-1]['content'],re.S)
                        label = json.loads(answer[1])
                        assert label == row['extra_info']['label']
                        assert label and set(label) <= TOOLS
                        assert all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and -100<=v<=100 for v in label.values())
                        assert row['images'] == [row['extra_info']['origin_image_path']]
                    else:
                        assert re.search(r'<answer>\s*(.*?)\s*</answer>',conv[-1]['content'],re.S)[1].strip() == row['extra_info']['label'].strip()
                except Exception as e:
                    errors.append(dict(id=row['id'],source_id=row['source_id'],error=f'{type(e).__name__}: {e}'))
                normalized.append(row)
            datasets[key]=normalized
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        images=dict(pool.map(image_info,sorted(paths)))
    bad_images={p:v for p,v in images.items() if 'error' in v}
    dump(OUT/'image-manifest.json',images)
    dump(OUT/'invalid-records.json',dict(labels=errors,images=bad_images))
    if errors or bad_images:
        raise RuntimeError(f'Dataset validation failed: {len(errors)} records, {len(bad_images)} images')
    test_origins = {r['extra_info']['origin_image_path'] for key,ds in datasets.items() if key.endswith('/test') for r in ds}
    test_hashes = {images[p]['pixel_sha256'] for p in test_origins}
    train_origins = {r['extra_info']['origin_image_path'] for key,ds in datasets.items() if key.endswith('/train') for r in ds}
    ready_stats, exclusions = {}, []
    for key, ds in datasets.items():
        task,split=key.split('/')
        dump(OUT/'normalized'/task/f'sft_{split}.json',ds)
        ready=[]
        for row in ds:
            origin=row['extra_info']['origin_image_path']
            if split=='train' and images[origin]['pixel_sha256'] in test_hashes:
                exclusions.append(dict(id=row['id'],source_id=row['source_id'],task=task,
                                       origin=origin,reason='origin image also occurs in an official test task (same decoded RGB pixels)'))
            else:
                ready.append(row)
        dump(OUT/'ready'/task/f'sft_{split}.json',ready)
        ready_stats[key]=dict(rows=len(ready),unique_origins=len({r['extra_info']['origin_image_path'] for r in ready}),
                             removed_for_test_overlap=len(ds)-len(ready))
        if split=='train':
            assert not {images[r['extra_info']['origin_image_path']]['pixel_sha256'] for r in ready}&test_hashes
    dump(OUT/'excluded-training-rows.json',exclusions)
    registry_path=ROOT/'data/dataset_info.json'
    backup=ROOT/'data/dataset_info.before-stage1.json'
    if not backup.exists(): backup.write_bytes(registry_path.read_bytes())
    registry=json.loads(registry_path.read_text())
    for task,name in [('GIER-Edit','gier_sft_distill'),('GIER-Summary','gier_sft_summary')]:
        template=registry[name]
        for split,suffix in [('train',''),('test','_test')]:
            entry=copy.deepcopy(template)
            entry['file_name']=str(OUT/'ready'/task/f'sft_{split}.json')
            registry[name+suffix]=entry
    dump(registry_path,registry)
    report=dict(timestamp=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),source=source_stats,ready=ready_stats,
                total_checked_images=len(images),missing_or_corrupt_images=len(bad_images),invalid_labels=len(errors),
                source_images_requiring_metadata_repair=len(_REPAIRS),image_repairs=_REPAIRS,
                source_train_test_origin_path_overlap=len(train_origins&test_origins),
                excluded_training_rows=len(exclusions),ready_train_test_exact_pixel_overlap=0,
                caveats=['Validation split and 3000-sample training subset are not yet frozen.',
                         'Near-duplicate/perceptual similarity has not been audited.',
                         'Ready training data removes official test-source overlap; it differs from the original paper recipe.',
                         'Repeated official IDs are preserved in source_id; id is now unique per task/split/source row.'])
    dump(ROOT/'logs/stage1-data-readiness.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

"""Freeze Stage 2 inputs after conservative project-record usage exclusions. CPU only."""
import collections
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/mnt/sdc/zhc/IEA')
OUT = ROOT / 'data/stage2/stage2-A-edit-grpo-v1'
SEED = 20261101


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def main():
    assert not OUT.exists(), 'Existing split must be verified, never overwritten'
    old = ROOT / 'data/stage1/splits/image-edit-3k-v1'
    manifest = read(old / 'manifest.json')
    for entry in manifest['files'].values():
        assert sha(Path(entry['path'])) == entry['sha256']
    adapter = ROOT / 'outputs/stage1-A-image-edit-3k-v1/adapter_model.safetensors'
    assert sha(adapter) == 'c40552634bfa9a55f2d8a1584361331b7c17b753a46f1f5efa659de459e16306'
    groups = read(old / 'group-assignments.json')
    path_group = {p: g['group_id'] for g in groups for p in g['source_paths']}
    train = read(old / 'train.json')
    used = {r['group_id'] for r in train}
    candidates = {g['group_id'] for g in groups if g['partition'] == 'training_pool'} - used
    evidence = []
    # Exclude every recorded image/ID in training, benchmarks, reload probes and inference.
    # Full-corpus integrity/token audits are preprocessing, not model development.
    def scan(value):
        if isinstance(value, dict):
            for v in value.values():
                yield from scan(v)
        elif isinstance(value, list):
            for v in value:
                yield from scan(v)
        elif isinstance(value, str):
            yield value
    ready = ROOT / 'data/stage1/ready'
    all_rows = []
    for task in ['GIER-Edit', 'GIER-Summary']:
        for split in ['train', 'test']:
            all_rows.extend(read(ready / task / f'sft_{split}.json'))
    id_group = {r['id']: path_group.get(r['extra_info']['origin_image_path']) for r in all_rows}
    paths = set((ROOT / 'logs').rglob('*.json'))
    paths |= set((ROOT / 'outputs/evaluation-v1').rglob('run.json'))
    paths |= set((ROOT / 'outputs/evaluation-v1').rglob('predictions.jsonl'))
    paths |= set((ROOT / 'data/stage1/splits').rglob('*train*.json'))
    paths.add(ROOT / 'data/smoke.json')
    # The readiness script preprocesses first four rows of each task, including Summary.
    for task in ['GIER-Edit', 'GIER-Summary']:
        rows = read(ready / task / 'sft_train.json')[:4]
        matched = {path_group.get(r['extra_info']['origin_image_path']) for r in rows} - {None}
        used |= matched
        evidence.append(dict(source=f'readiness-first-four:{task}', groups=sorted(matched)))
    for p in sorted(paths):
        if p.name in {'frozen-sequence-audit.json', 'stage1-data-readiness.json'}:
            continue
        values = [json.loads(x) for x in p.read_text().splitlines()] if p.suffix == '.jsonl' else [read(p)]
        strings = set(scan(values))
        matched = {path_group[s] for s in strings if s in path_group}
        matched |= {id_group[s] for s in strings if id_group.get(s)}
        matched |= strings & {g['group_id'] for g in groups}
        used |= matched
        evidence.append(dict(source=str(p), sha256=sha(p), matched_groups=sorted(matched)))
    eligible = sorted(candidates - used)
    assert len(eligible) >= 100, (len(candidates), len(eligible))
    rng = random.Random(SEED)
    by_type = collections.defaultdict(list)
    for r in train:
        by_type[r['extra_info']['inst_type']].append(r)
    targets = {k: len(v) * 512 / len(train) for k, v in by_type.items()}
    quotas = {k: int(v) for k, v in targets.items()}
    for k in sorted(targets, key=lambda k: (-(targets[k]-quotas[k]), k))[:512-sum(quotas.values())]:
        quotas[k] += 1
    selected = []
    for k in sorted(by_type):
        selected.extend(rng.sample(sorted(by_type[k], key=lambda r: r['id']), quotas[k]))
    rng.shuffle(selected)
    new_groups = set(rng.sample(eligible, 100))
    pool = collections.defaultdict(list)
    for r in read(ready / 'GIER-Edit/sft_train.json'):
        g = path_group[r['extra_info']['origin_image_path']]
        if g in new_groups:
            pool[g].append(dict(r, group_id=g))
    new_test = [rng.choice(sorted(pool[g], key=lambda r: r['id'])) for g in sorted(new_groups)]
    splits = dict(train=selected, preflight=selected[:32], validation=read(old/'validation.json'),
                  test_new=new_test, test_regression=read(old/'test.json'))
    for a, b in [('train','validation'), ('train','test_new'), ('train','test_regression'),
                 ('validation','test_new'), ('test_new','test_regression')]:
        assert not {r['group_id'] for r in splits[a]} & {r['group_id'] for r in splits[b]}
    from PIL import Image
    for p in sorted({r['extra_info'][k] for rows in splits.values() for r in rows
                     for k in ['origin_image_path', 'reference_image_path']}):
        with Image.open(p) as im:
            im.load()
    OUT.mkdir(parents=True)
    files = {}
    for name, rows in splits.items():
        p = OUT / f'{name}.json'
        p.write_text(json.dumps(rows, indent=2, ensure_ascii=False)+'\n')
        files[name] = dict(path=str(p), sha256=sha(p), rows=len(rows),
                           unique_groups=len({r['group_id'] for r in rows}))
    result = dict(status='frozen', created_at=datetime.now(timezone.utc).isoformat(), seed=SEED,
                  files=files, instruction_type_quotas=quotas, adapter_sha256=sha(adapter),
                  candidate_groups=len(candidates), historically_excluded_candidates=sorted(candidates & used),
                  eligible_new_test_groups=len(eligible), historical_usage_evidence=evidence,
                  limitations=['Project-record audit; cannot certify unrecorded manual/external usage or upstream pretraining.',
                               'All corpus rows underwent CPU integrity preprocessing; this is not untouched raw data.',
                               'No perceptual near-duplicate certification.'],
                  generation_input_rule='Only original image and system/user messages; assistant and reference excluded.')
    (OUT/'manifest.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'historical_usage_evidence'}, indent=2))


if __name__ == '__main__':
    main()

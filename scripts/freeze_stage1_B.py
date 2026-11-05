"""Freeze the planned sampling-only B control before any held-out test inference."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import yaml

ROOT = Path('/mnt/sdc/zhc/IEA')
SOURCE = ROOT / 'data/stage1/splits/image-edit-3k-v1'
DEST = ROOT / 'data/stage1/splits/image-edit-3k-B-v1'
CONFIG = ROOT / 'configs/qlora-stage1-B-3k.yaml'
MANIFEST = ROOT / 'configs/stage1-B-3k-manifest.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def active(row):
    return {key for key, value in row['extra_info']['label'].items() if value != 0}


def counts(rows):
    return dict(Counter(tool for row in rows for tool in active(row)))


def main():
    subprocess.run([str(ROOT / 'scripts/train_stage1_frozen.sh'), '--check-only'], check=True)
    assert not DEST.exists() and not CONFIG.exists() and not MANIFEST.exists()
    rows = json.loads((SOURCE / 'train.json').read_text())
    assert len(rows) == 3000
    freq = counts(rows)
    rare = sorted(tool for tool, count in freq.items() if count / len(rows) < .25)
    assert rare == ['exposure', 'sharpness'], rare
    weights = [2 if active(row) & set(rare) else 1 for row in rows]
    seed = 20261031
    draw = random.Random(seed).choices(range(len(rows)), weights=weights, k=3000)
    sampled = []
    for i, index in enumerate(draw):
        row = deepcopy(rows[index])
        row['sampling_source_id'] = row['id']
        row['id'] = f'GIER-Edit:B:draw:{i:06d}'
        sampled.append(row)
    audit_path = SOURCE / 'sequence-lengths-262k.json'
    audit = {row['id']: row for row in json.loads(audit_path.read_text())['train']}
    assert set(audit) == {row['id'] for row in rows}
    assert not any(row['truncated'] for row in audit.values())
    def token_exposures(selected):
        return {key: 2 * sum(audit[rows[i]['id']][key] for i in selected)
                for key in ('tokens', 'supervised_tokens', 'image_tokens')}
    DEST.mkdir(parents=True)
    write(DEST / 'train.json', sampled)
    write(DEST / 'draw-ids.json', [dict(id=row['id'], source_id=row['sampling_source_id'],
                                     group_id=row['group_id']) for row in sampled])
    base = yaml.safe_load((ROOT / 'configs/qlora-stage1-A-3k.yaml').read_text())
    b = deepcopy(base)
    b.update(dataset='iea_edit_3k_B_train', output_dir=str(ROOT / 'outputs/stage1-B-image-edit-3k-v1'))
    CONFIG.write_text(yaml.safe_dump(b, sort_keys=False))
    registry_path = ROOT / 'data/dataset_info.json'
    registry = json.loads(registry_path.read_text())
    assert 'iea_edit_3k_B_train' not in registry
    registry['iea_edit_3k_B_train'] = deepcopy(registry['iea_edit_3k_train'])
    registry['iea_edit_3k_B_train']['file_name'] = str(DEST / 'train.json')
    write(registry_path.with_suffix('.tmp'), registry)
    registry_path.with_suffix('.tmp').replace(registry_path)
    decision = json.loads((ROOT / 'configs/stage1-A-3k-decision.json').read_text())
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), version='image-edit-3k-B-v1',
                    hypothesis='Doubling selection weight for examples with a nonzero rare tool may improve rare-tool behavior; assess both the predeclared subgroup and overall regression.',
                    sampling=dict(seed=seed, rare_tool_threshold_fraction=.25, rare_tools=rare,
                                  weight_with_any_rare_tool=2, weight_other=1,
                                  method='Python random.Random(seed).choices with replacement; draw 3000 rows once and traverse this fixed list twice'),
                    source_train_path=str(SOURCE / 'train.json'), source_train_sha256=sha(SOURCE / 'train.json'),
                    train_path=str(DEST / 'train.json'), train_sha256=sha(DEST / 'train.json'),
                    draw_ids_sha256=sha(DEST / 'draw-ids.json'),
                    train_rows=3000, unique_source_rows=len(set(draw)),
                    unique_source_groups=len({row['group_id'] for row in sampled}),
                    source_nonzero_tool_counts=freq, sampled_nonzero_tool_counts=counts(sampled),
                    source_instruction_types=dict(Counter(row['extra_info']['inst_type'] for row in rows)),
                    sampled_instruction_types=dict(Counter(row['extra_info']['inst_type'] for row in sampled)),
                    A_token_exposures=token_exposures(range(3000)), B_token_exposures=token_exposures(draw),
                    token_audit_sha256=sha(audit_path),
                    config_path=str(CONFIG), config_sha256=sha(CONFIG),
                    allowed_config_differences=['dataset', 'output_dir'],
                    source_A_config_sha256=decision['formal_config_sha256'],
                    split_manifest_sha256=decision['split_manifest_sha256'],
                    model_revision=decision['model_revision'], repository_commit=decision['repository_commit'],
                    validation_dataset='iea_edit_3k_validation', optimizer_steps=376, test_inference_started=False,
                    predeclared_subgroup='Test rows whose distilled nonzero target tools include exposure or sharpness; not a subgroup selected from model predictions.',
                    limitations=['Fixed-budget resampling changes unique-example coverage and token exposure; equal steps are not exactly equal computation.',
                                 'Rare tool use is defined by distilled targets and is not guaranteed to coincide with the wording of the user instruction.',
                                 'This is one exploratory single-seed control, frozen after A training but before test inference.'])
    write(MANIFEST, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

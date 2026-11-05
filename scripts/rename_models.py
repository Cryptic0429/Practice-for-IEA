"""Give delivered models descriptive names that expose their training parent.

Python 3.8 compatible. Model/config bytes and historical experiment identities
remain unchanged; original model paths become compatibility links.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request


RENAMES = [
    ('A', 'models/stage1/A', 'models/from-base/sft-standard', '标准监督训练', 20261031),
    ('B', 'models/stage1/B', 'models/from-base/sft-resampled', '重采样监督训练', 20261031),
    ('SFT', 'models/stage2/SFT', 'models/from-sft-standard/sft-continued', '继续监督训练', 20261101),
    ('GRPO', 'models/stage2/GRPO', 'models/from-sft-standard/grpo-l1', 'L1 奖励强化学习', 20261101),
    ('Seed20261103', 'models/quality-v2-20261103/Seed20261103',
     'models/from-sft-standard/sft-calibrated/seed-01', '校准监督训练·复现1', 20261103),
    ('Seed20261104', 'models/quality-v2-20261103/Seed20261104',
     'models/from-sft-standard/sft-calibrated/seed-02', '校准监督训练·复现2', 20261104),
    ('Seed20261105', 'models/quality-v2-20261103/Seed20261105',
     'models/from-sft-standard/sft-calibrated/seed-03', '校准监督训练·复现3', 20261105),
]
SELECTED = 'models/from-sft-standard/sft-calibrated/seed-02'
BASE_ID = 'qwen2.5-vl-7b-instruct'
BASE_REVISION = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_new(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def atomic_link(path, target):
    assert (path.parent / target).exists() if not os.path.isabs(target) else Path(target).exists()
    tmp = path.with_name('.' + path.name + '.naming-tmp')
    assert not tmp.exists() and not tmp.is_symlink()
    tmp.symlink_to(target, target_is_directory=True)
    os.replace(tmp, path)


def relocate(root, old_name, new_name):
    old, new = root / old_name, root / new_name
    assert old.is_dir() and not new.exists() and not new.is_symlink()
    new.parent.mkdir(parents=True, exist_ok=True)
    resolved = old.resolve()
    if old.is_symlink():
        # The server's model directory may already be a link into a frozen run.
        new.symlink_to(str(resolved), target_is_directory=True)
        atomic_link(old, str(new))
        operation = 'descriptive_entry_and_legacy_alias'
    else:
        assert not any(p.is_symlink() for p in old.rglob('*'))
        inode = old.stat().st_ino
        old.rename(new)
        old.symlink_to(str(new), target_is_directory=True)
        assert new.stat().st_ino == inode
        operation = 'rename_and_legacy_alias'
    assert old.resolve() == new.resolve()
    return {'old': old_name, 'new': new_name, 'operation': operation}


def model_info(root, item, lock, base_shards):
    old_id, old_path, new_path, display, seed = item
    spec = lock['models'][old_id]
    info = {
        'schema': 'iea-model-info-v1',
        'id': new_path[len('models/'):],
        'display_name': display,
        'legacy_id': old_id,
        'canonical_path': new_path,
        'kind': 'lora_adapter',
        'base_model': {
            'id': BASE_ID, 'upstream': 'Qwen/Qwen2.5-VL-7B-Instruct',
            'revision': BASE_REVISION,
            'server_path': '/mnt/sdc/zhc/IEA/models/Qwen2.5-VL-7B-Instruct'},
        'adapter_sha256': spec['adapter_sha256'],
        'adapter_config_sha256': spec['config_sha256'],
        'training_seed': seed,
        'selected_for_current': old_id == 'Seed20261104',
        'evaluation_records': [
            'experiments/quality-v2-20261103/metadata/final-models-lock.json',
            'experiments/quality-v2-20261103/metadata/final-analysis-summary.json',
            'experiments/quality-v2-20261103/review/visual-review-summary.json'],
        'availability': {'local': True, 'server': True},
        'name_changed_on': '2026-11-04',
    }
    if old_id in ('A', 'B'):
        info['parent_model'] = {'id': BASE_ID, 'legacy_id': 'Base',
                                'weight_shards_sha256': base_shards}
        summary = read(root / ('experiments/stage1/training-' + old_id + '/summary.json'))
        info['training'] = {
            'method': 'QLoRA supervised fine-tuning', 'rows_or_draws': 3000,
            'optimizer_steps': 376, 'epochs': 2, 'learning_rate': 0.0001,
            'target_format': 'explanation and fenced parameter JSON',
            'started_at': summary.get('started_at'),
            'completion_confirmed_at': summary.get('completion_confirmed_at'),
            'configuration': 'configs/qlora-stage1-' + old_id + '-3k.yaml',
            'sampling': 'original training distribution' if old_id == 'A' else
                        'exposure/sharpness weighted resampling with replacement',
            'source_groups': 1078 if old_id == 'A' else summary.get('unique_source_groups'),
        }
        info['export'] = {'method': 'validation-selected checkpoint', 'checkpoint_step': 376}
    else:
        info['parent_model'] = {
            'id': 'from-base/sft-standard', 'legacy_id': 'A',
            'canonical_path': 'models/from-base/sft-standard',
            'adapter_sha256': lock['models']['A']['adapter_sha256']}
        if old_id in ('SFT', 'GRPO'):
            info['training'] = {
                'method': 'continued supervised fine-tuning' if old_id == 'SFT' else 'GRPO',
                'rows': 512, 'source_groups': 414, 'optimizer_steps': 128,
                'learning_rate': 0.000001,
                'dataset_server_path': '/mnt/sdc/zhc/IEA/data/stage2/stage2-A-edit-grpo-v1/train.json',
                'dataset_sha256': 'de565c842980de3a1c0f66154cec602be42288f5574b35ae5f7d5d41e9ced5fc',
                'evidence': 'experiments/stage2/metadata/data-manifest.json',
            }
            if old_id == 'GRPO':
                info['training']['reward'] = 'historical simplified pixel-L1 improvement reward'
                info['training']['candidates_per_prompt'] = 4
            info['export'] = {'method': 'final checkpoint' if old_id == 'SFT' else
                              'validation-selected checkpoint', 'checkpoint_step': 128}
        else:
            info['training'] = {
                'method': 'continued QLoRA SFT on calibrated labels; JSON-only supervision',
                'rows': 3000, 'source_groups': 1078, 'optimizer_steps': 1024,
                'sample_exposures': 4096, 'learning_rate': 0.0001,
                'dataset': 'data/training/quality-v2-20261103/train3k-contrast-calibrated.json',
                'dataset_sha256': 'd1dc31ae91f583e1d672ba047d390723d3b1ce6494ea88f802bb67356a226a19',
                'evidence': 'experiments/quality-v2-20261103/metadata/development-training-lock-3k-v2.json',
                'timestamp_note': 'Exact run times, when recorded, remain in the historical run logs; seed values are not dates.'}
            info['export'] = {'method': 'LoRA A/B tensor averaging within one training trajectory',
                              'checkpoint_steps': [512, 1024], 'weights': [0.5, 0.5]}
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--mode', choices=['local', 'server'], required=True)
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    root = a.root.resolve()
    lock_path = root / 'experiments/quality-v2-20261103/metadata/final-models-lock.json'
    lock = read(lock_path)
    evaluations = read(root / 'configs/evaluation-models-v1.json')['models']
    base_shards = {Path(s['path']).name: s['sha256'] for s in evaluations['Base']['weight_shards']}
    infos = [model_info(root, item, lock, base_shards) for item in RENAMES]
    # Validate the complete proposal before any model location is changed.
    for item, info in zip(RENAMES, infos):
        source, dest = root / item[1], root / item[2]
        assert source.is_dir() and not dest.exists() and not dest.is_symlink(), item
        assert not (source / 'model-info.json').exists()
        assert sha(source / 'adapter_model.safetensors') == info['adapter_sha256']
        assert sha(source / 'adapter_config.json') == info['adapter_config_sha256']
    if not a.apply:
        print(json.dumps({'mode': a.mode, 'models': len(infos),
                          'renames': [{'legacy': i[0], 'new': i[2]} for i in RENAMES],
                          'selected': SELECTED}, ensure_ascii=False, indent=2))
        return
    audit_dir = root / 'archives/model-naming-20261104'
    audit_dir.mkdir(parents=True, exist_ok=True)
    receipt = audit_dir / (a.mode + '-renaming.json')
    assert not receipt.exists()
    backup = audit_dir / 'before-documents/README.md'
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / 'README.md', backup)
    protected = {str(lock_path.relative_to(root)): sha(lock_path)}
    for name in lock['code_hashes']:
        if (root / name).exists():
            protected[name] = sha(root / name)
    report = {'status': 'in_progress', 'mode': a.mode, 'operations': [],
              'changed_on': '2026-11-04', 'parent_model_is_visible_in_directory_names': True}

    def save():
        tmp = receipt.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(receipt)

    save()
    for item, info in zip(RENAMES, infos):
        report['operations'].append(relocate(root, item[1], item[2]))
        write_new(root / item[2] / 'model-info.json', info)
        save()
    # The stage-2 A entry is a reference copy of the stage-1 parent, not a new model.
    old_a = root / 'models/stage2/A'
    assert sha(old_a / 'adapter_model.safetensors') == lock['models']['A']['adapter_sha256']
    assert sha(old_a / 'adapter_config.json') == lock['models']['A']['config_sha256']
    if not old_a.is_symlink():
        archived = audit_dir / 'legacy-copies/stage2-A'
        archived.parent.mkdir(parents=True, exist_ok=True)
        old_a.rename(archived)
        old_a.symlink_to(str(root / 'models/from-base/sft-standard'), target_is_directory=True)
        report['stage2_A_duplicate_preserved_at'] = str(archived.relative_to(root))
    else:
        atomic_link(old_a, str(root / 'models/from-base/sft-standard'))
    # Consolidate compatibility entries so the new names are the primary model view.
    for name in ['stage1', 'stage2', 'quality-v2-20261103']:
        old = root / 'models' / name
        new = root / 'models/legacy' / name
        new.parent.mkdir(parents=True, exist_ok=True)
        assert old.is_dir() and not old.is_symlink() and not new.exists()
        old.rename(new)
        old.symlink_to(str(new), target_is_directory=True)
        report['operations'].append({'old': str(old.relative_to(root)),
                                     'new': str(new.relative_to(root)), 'operation': 'group_legacy_aliases'})
    atomic_link(root / 'models/current', SELECTED[len('models/'):])
    catalog = {'schema': 'iea-model-catalog-v1', 'changed_on': '2026-11-04',
               'current': SELECTED[len('models/'):],
               'lineage': {BASE_ID: ['from-base/sft-standard', 'from-base/sft-resampled'],
                           'from-base/sft-standard': [i['id'] for i in infos if i['legacy_id'] not in ('A', 'B')]},
               'models': infos,
               'full_models': [
                   {'id': BASE_ID, 'display_name': 'Qwen2.5-VL-7B-Instruct',
                    'server_path': '/mnt/sdc/zhc/IEA/models/Qwen2.5-VL-7B-Instruct',
                    'revision': BASE_REVISION, 'availability': {'local': False, 'server': True}},
                   {'id': 'official-iea', 'display_name': 'Official IEA',
                    'server_path': '/mnt/sdc/zhc/IEA/models/Qwen2.5-VL-7B-IEA',
                    'availability': {'local': False, 'server': True}}],
               'historical_identifiers': 'Frozen reports, model locks, and blind-review keys retain their original identifiers.'}
    if a.mode == 'server':
        for entry in catalog['full_models']:
            write_new(Path(entry['server_path']) / 'model-info.json', entry)
        runtime = root / 'models/Qwen2.5-VL-7B-IEA-runtime'
        write_new(runtime / 'model-info.json', {
            'id': 'official-iea-runtime', 'display_name': 'Official IEA 兼容加载目录',
            'kind': 'runtime_compatibility_directory', 'parent_model': {'id': 'official-iea'},
            'weight_policy': 'Shares the published Official IEA weight files; not a separately trained model.'})
        extra = root / 'models/goal-v2-A-expanded-start-v1'
        if extra.exists():
            new_extra = 'models/from-sft-standard/diagnostics/expanded-lora-start'
            report['operations'].append(relocate(root, str(extra.relative_to(root)), new_extra))
            verification = read(root / new_extra / 'start-verification.json')
            write_new(root / new_extra / 'model-info.json', {
                'id': new_extra[len('models/'):], 'display_name': '标准监督训练的扩展 LoRA 诊断起点',
                'kind': 'diagnostic_initialization',
                'parent_model': {'id': 'from-base/sft-standard', 'legacy_id': 'A',
                                 'adapter_sha256': verification['original_A_sha256']},
                'adapter_sha256': verification['adapter_sha256'],
                'training': {'optimizer_updates': verification.get('optimizer_updates', 0)},
                'selected_for_current': False})
            save()
    write_new(root / 'models/catalog.json', catalog)
    table = '\n'.join('| {} | {} | {} |'.format(
        i['legacy_id'], '[' + i['display_name'] + '](' + i['id'] + ')',
        '基础 Qwen' if i['legacy_id'] in ('A', 'B') else '标准监督训练（原 A）') for i in infos)
    text = '''# IEA 模型名称与训练谱系

`from-base/` 表示从原始 Qwen 开始训练；`from-sft-standard/` 表示从标准监督训练模型（原 A）继续训练。

| 历史标识 | 新名称与目录 | 训练起点 |
|---|---|---|
''' + table + '''

当前选定入口是 [current](current)，指向校准监督训练·复现2。
每个模型的 `model-info.json` 记录父模型名称、父权重身份、数据、实际随机种子和导出方式；完整机器清单见 [catalog.json](catalog.json)。三个 `seed-01/02/03` 是复现实验编号，实际种子分别为 20261103、20261104、20261105。

历史实验、冻结模型锁及匿名图对应表继续使用 A/B/SFT/GRPO/Seed 标识。`legacy/` 集中保存旧名称入口，旧 `stage1/`、`stage2/`、`quality-v2-20261103/` 路径仍可访问同一模型。

完整基础模型 Qwen2.5-VL-7B-Instruct 和作者模型 Official IEA 保留在服务器原目录；本地仅保存自训练 adapter。Official runtime 为兼容加载目录。
'''
    (root / 'models/README.md').write_text(text)
    readme = (root / 'README.md').read_text()
    readme = readme.replace('项目现在只保留三份 Markdown，按下面的顺序阅读即可。',
                            '项目的三份主文档按下面的顺序阅读即可。模型名称与训练谱系另见 [模型目录说明](models/README.md)。')
    readme = readme.replace('交付模型为 **Seed20261104**', '交付模型为 **校准监督训练·复现2（原 Seed20261104）**')
    readme = readme.replace('| [当前模型](models/current) | Seed20261104 adapter；基础权重仍需从服务器固定版本加载 |',
                            '| [当前模型](models/current) | 校准监督训练·复现2；从标准监督训练（原 A）继续训练；完整底座在服务器 |')
    readme = readme.replace('| [全部 adapter](models) | Stage 1、Stage 2、质量优化三种子分组保存 |',
                            '| [模型名称与训练谱系](models/README.md) | 按训练起点分组：from-base 与 from-sft-standard；每个模型附 model-info.json |')
    tmp = root / '.README.model-naming.tmp'
    tmp.write_text(readme)
    tmp.replace(root / 'README.md')
    verified = []
    for item, info in zip(RENAMES, infos):
        new = root / item[2]
        assert sha(new / 'adapter_model.safetensors') == info['adapter_sha256']
        assert sha(new / 'adapter_config.json') == info['adapter_config_sha256']
        assert (root / item[1]).resolve() == new.resolve()
        assert read(new / 'model-info.json')['parent_model'] == info['parent_model']
        verified.append(info['id'])
    assert (root / 'models/current').resolve() == (root / SELECTED).resolve()
    for name, before in protected.items():
        assert sha(root / name) == before, name
    report.update(status='completed', verified_models=verified, current=SELECTED,
                  protected_frozen_files_unchanged=len(protected),
                  catalog_sha256=sha(root / 'models/catalog.json'))
    if a.mode == 'server':
        with urllib.request.urlopen('http://127.0.0.1:12345/', timeout=8) as response:
            report['demo_http_status'] = response.status
        assert report['demo_http_status'] == 200
    save()
    print(json.dumps({'status': report['status'], 'mode': a.mode,
                      'renamed_models': len(verified), 'current': SELECTED,
                      'protected_frozen_files_unchanged': len(protected)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

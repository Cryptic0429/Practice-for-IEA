"""Freeze the evaluation protocol and later lock all selected models before test inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

ROOT = Path('/mnt/sdc/zhc/IEA')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--models', action='store_true')
    args = cli.parse_args()
    protocol_path = ROOT / 'configs/evaluation-protocol-v1.json'
    p = read(protocol_path)
    assert not any((d / 'predictions.jsonl').exists() for d in (ROOT / 'outputs/evaluation-v1').glob('test-*'))
    if args.models:
        assert p['status'] == 'frozen'
        out = ROOT / 'configs/evaluation-models-v1.json'
        assert not out.exists()
        integrity_path = ROOT / 'configs/evaluation-weight-integrity-v1.json'
        integrity = read(integrity_path)
        models = {}
        for model, spec in p['models'].items():
            files = [Path(spec['model_name_or_path']) / name for name in ('config.json', 'model.safetensors.index.json', 'tokenizer_config.json')]
            files += [Path(spec['model_name_or_path']) / name for name in
                      ('preprocessor_config.json', 'processor_config.json', 'tokenizer.json',
                       'vocab.json', 'merges.txt', 'chat_template.json', 'generation_config.json')
                      if (Path(spec['model_name_or_path']) / name).is_file()]
            entry = dict(model_name_or_path=spec['model_name_or_path'])
            if model in ('A', 'B'):
                summary = read(ROOT / f'logs/stage1-{model}-3k/summary.json')
                reload_report = read(ROOT / f'logs/stage1-{model}-3k/reload-verification.json')
                assert summary['status'] == 'completed_and_reload_verified' and reload_report['adapter_reload_verified']
                assert summary['adapter_sha256'] == reload_report['adapter_sha256']
                adapter = Path(spec['adapter_name_or_path'])
                files += [adapter / 'adapter_model.safetensors', adapter / 'adapter_config.json']
                assert sha(adapter / 'adapter_model.safetensors') == summary['adapter_sha256']
                entry.update(adapter_name_or_path=str(adapter), best_checkpoint=summary['best_checkpoint'],
                             best_validation_loss=summary['best_validation_loss'])
            entry['identity_files'] = [dict(path=str(path), sha256=sha(path)) for path in files]
            # Weight shard metadata catches accidental replacement after selection; original Qwen
            # full SHA-256 verification is separately retained in preparation records.
            directory = Path(spec['model_name_or_path'])
            entry['weight_shards'] = [dict(path=str(path), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
                                      for path in sorted(directory.glob('*.safetensors'))]
            assert entry['weight_shards']
            family = 'Official' if model == 'Official' else 'Base'
            content_hashes = {item['path']:item for item in integrity['models'][family]['shards']}
            assert set(content_hashes) == {item['path'] for item in entry['weight_shards']}
            for item in entry['weight_shards']:
                proof = content_hashes[item['path']]
                assert (item['bytes'], item['mtime_ns']) == (proof['bytes'], proof['mtime_ns'])
                item['sha256'] = proof['sha256']
            entry['identity_files'].append(dict(path=str(integrity_path), sha256=sha(integrity_path)))
            models[model] = entry
        write(out, dict(frozen_at=datetime.now(timezone.utc).isoformat(), protocol_sha256=sha(protocol_path),
                        all_selected_models_verified=True, models=models))
        print('ALL_EVALUATION_MODELS_LOCKED', out)
        return
    assert p['status'] == 'draft_for_train_only_engineering'
    for model in ['Base', 'A', 'Official', 'BaseSchema']:
        candidates = list((ROOT / 'outputs/evaluation-v1').glob(f'train-{model}-engineering-*/status.json'))
        completed = [read(path) for path in candidates if read(path)['status'] == 'completed']
        assert completed and any(s['completed'] == 8 and s['inference_errors'] == 0 for s in completed), model
    source = Path(p['split_directory'])
    manifest = read(source / 'manifest.json')
    for name in ['train', 'validation', 'test']:
        assert sha(source / f'{name}.json') == manifest['files'][name]['sha256']
    # IDs are selected from the existing fixed list without observing any model predictions.
    test = read(source / 'test.json')
    chosen = random.Random(p['seed']).sample([row['id'] for row in test], 30)
    permutation_rng = random.Random(p['seed'] + 1)
    key = []
    for sample_id in chosen:
        models = list(p['models'])
        permutation_rng.shuffle(models)
        key.append(dict(id=sample_id, panels={f'Model {i+1}': model for i, model in enumerate(models)}))
    folder = ROOT / 'data/stage1/evaluation-v1'
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / 'visual-ids.json', chosen)
    write(folder / 'visual-blinding-key.json', key)
    code = ['scripts/evaluation_core.py', 'scripts/evaluate_stage1.py', 'scripts/summarize_evaluation.py',
            'repo/src/image_editor.py', 'repo/src/utils.py', 'repo/src/config.py',
            'repo/LLaMA-Factory/src/llamafactory/chat/hf_engine.py',
            'repo/LLaMA-Factory/src/llamafactory/data/template.py',
            'repo/LLaMA-Factory/src/llamafactory/data/mm_plugin.py']
    p.update(status='frozen', frozen_at=datetime.now(timezone.utc).isoformat(),
             code_hashes={str(ROOT / rel): sha(ROOT / rel) for rel in code},
             split_manifest_sha256=sha(source / 'manifest.json'),
             visual_ids_path=str(folder / 'visual-ids.json'), visual_ids_sha256=sha(folder / 'visual-ids.json'),
             visual_blinding_key_sha256=sha(folder / 'visual-blinding-key.json'))
    p['official_scope'] += ' The source-data audit found original-image overlap in the authors\' train/test split; Official may have seen sources of this held-out subset and is not a clean causal control.'
    write(protocol_path, p)
    print('EVALUATION_PROTOCOL_FROZEN', protocol_path, sha(protocol_path))


if __name__ == '__main__':
    main()

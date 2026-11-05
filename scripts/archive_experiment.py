"""Build a checksummed acceptance bundle after every required experiment is complete.

The bundle contains adapters, frozen splits, predictions, reports and selected
visual assets. Large base weights, the full image corpus and optimizer states
remain at the documented server paths; this is not a self-contained data release.
"""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path('/mnt/sdc/zhc/IEA')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    evaluation = ROOT / 'outputs/evaluation-v1'
    protocol = read(ROOT / 'configs/evaluation-protocol-v1.json')
    lock = read(ROOT / 'configs/evaluation-models-v1.json')
    assert lock['all_selected_models_verified']
    assert digest(ROOT / 'configs/evaluation-protocol-v1.json') == lock['protocol_sha256']
    summary = read(evaluation / 'summary.json')
    assert summary['sample_count'] == summary['independent_source_groups'] == 200
    files = set()

    def add(path):
        path = ROOT / path if not Path(path).is_absolute() else Path(path)
        assert path.is_file(), f'Missing required archive input: {path}'
        assert path.is_relative_to(ROOT)
        files.add(path)

    for name in ['README.md', '运行_评测与验收.md', '验收_T4_模型A正式训练.md', '验收_T4_模型B重采样对照.md',
                 '验收_T4-T7_完整实验报告.md', '诊断_T5_验证集失败分析.md', 'IEA_SJTU1_两天任务表与规划.md']:
        add(name)
    for name in ['qlora-stage1-A-3k.yaml', 'qlora-stage1-B-3k.yaml', 'stage1-A-3k-decision.json',
                 'stage1-B-3k-manifest.json', 'evaluation-protocol-v1.json', 'evaluation-models-v1.json', 'evaluation-weight-integrity-v1.json',
                 'installed-packages.txt', 'report-installed-packages.txt', 'constraints.txt',
                 'qwen-base-revision.json', 'official-eval-compatibility.json', 'stage1-assets.json',
                 'repository-state.json', 'activation-template.sh']:
        add(Path('configs') / name)
    for source in (ROOT / 'scripts').iterdir():
        if source.suffix in ('.py', '.sh'):
            add(source)
    for name in ['queue-launch.json', 'queue-status.json', 'queue-events.jsonl', 'queue.log']:
        add(Path('logs/evaluation-v1') / name)
    for path in (ROOT / 'logs/evaluation-v1').glob('prep*'):
        if path.is_file() and path.suffix in ('.json', '.jsonl', '.log'):
            add(path)
    add('logs/evaluation-v1/input-tensor-parity-test-audit.log')
    for name in ['image_editor.py', 'config.py', 'utils.py']:
        add(Path('repo/src') / name)
    # Preserve the evidence linked by the project documents. These are copied
    # from the Mac workspace before archiving; omit large download indexes.
    metadata = ROOT / 'metadata'
    for name in ['stage1-assets.json', 'stage1-data-readiness.json',
                 'qwen-stage1-readiness.json', 'environment-readiness.json',
                 'stage1-A-3k-decision.json', 'stage1-B-3k-manifest.json']:
        add(metadata / name)
    for path in metadata.rglob('*'):
        if path.is_file() and 'archive-index' not in path.name:
            add(path)
    for name in ['train.json', 'validation.json', 'test.json', 'manifest.json']:
        add(Path(protocol['split_directory']) / name)
    add('data/stage1/splits/image-edit-3k-B-v1/train.json')
    add('data/dataset_info.json')
    for model in ['A', 'B']:
        log = ROOT / f'logs/stage1-{model}-3k'
        training = read(log / 'summary.json')
        assert training['status'] == 'completed_and_reload_verified' and training['global_step'] == 376
        adapter = ROOT / f'outputs/stage1-{model}-image-edit-3k-v1'
        assert digest(adapter / 'adapter_model.safetensors') == training['adapter_sha256']
        for name in ['run.json', 'status.json', 'events.jsonl', 'gpu.csv', 'train.log', 'summary.json',
                     'artifact-verification.json', 'reload-verification.json']:
            add(log / name)
        for name in ['trainer_state.json', 'trainer_log.jsonl', 'train_results.json', 'eval_results.json',
                     'adapter_model.safetensors', 'adapter_config.json']:
            add(adapter / name)
        for path in adapter.glob('*.json'):
            add(path)
    expected_ids = [r['id'] for r in read(Path(protocol['split_directory']) / 'test.json')]
    for model in protocol['models']:
        folder = evaluation / f'test-{model}-full'
        status = read(folder / 'status.json')
        assert status['status'] == 'completed' and status['completed'] == 200
        rows = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
        assert [row['id'] for row in rows] == expected_ids
        assert summary['models'][model]['n'] == 200
    # Preserve unsuccessful engineering runs and all model text, not only successes.
    for folder in evaluation.iterdir():
        if folder.is_dir() and folder.name.startswith(('train-', 'validation-', 'test-')):
            for path in folder.iterdir():
                if path.is_file() and path.suffix in ('.json', '.jsonl', '.log', '.py'):
                    add(path)
    for name in ['summary.json', 'metrics.csv', 'paired-rows.jsonl', 'validation-diagnostics.json',
                 'validation-target-diagnostics.json', 'instruction-display-audit.json',
                 'input-parity-audit.json', 'input-tensor-parity-audit.json', 'input-tensor-parity-test-audit.json', 'visual-review-observations.json',
                 'visual-review-decoded.json', 'figure-audit-summary.json', 'final-visual-qa.json', 'final-acceptance.json']:
        add(evaluation / name)
    acceptance = read(evaluation / 'final-acceptance.json')
    assert acceptance['status'] == 'accepted' and all(item['passed'] for item in acceptance['requirements'])
    observations = read(evaluation / 'visual-review-observations.json')
    assert len(observations['cases']) == 30
    assert [row['id'] for row in observations['cases']] == read(Path(protocol['visual_ids_path']))
    decoder = read(evaluation / 'visual-review-decoded.json')
    assert decoder['observations_sha256'] == digest(evaluation / 'visual-review-observations.json')
    for folder_name in ['blinded-visual-bundle', 'blinded-visual-render', 'result-figures',
                        'engineering-visual-preview', 'training-figures']:
        folder = evaluation / folder_name
        assert folder.is_dir()
        for path in folder.rglob('*'):
            if path.is_file():
                add(path)
    for path in (ROOT / 'data/stage1/evaluation-v1').glob('*.json'):
        add(path)
    add(evaluation / 'blinded-visual-bundle-private-provenance.json')
    entries = [dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=digest(path))
               for path in sorted(files)]
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), files=entries,
                    scope='Server replay and acceptance bundle. Base weights, original image corpus, full per-row edited PNGs outside the selected 30 cases, and optimizer states remain on SJTU1.',
                    base_model_revision=read(ROOT/'logs/stage1-A-3k/summary.json')['base_model_revision'],
                    external_assets=dict(base_model=str(ROOT/'models/Qwen2.5-VL-7B-Instruct'),
                                         official_model=str(ROOT/'models/Qwen2.5-VL-7B-IEA'),
                                         original_images=str(ROOT/'repo/datasets'),
                                         all_edited_outputs=str(evaluation),
                                         optimizer_states=[str(ROOT/f'outputs/stage1-{m}-image-edit-3k-v1') for m in ['A','B']]))
    output = ROOT / 'outputs/acceptance-v1'
    output.mkdir(exist_ok=False)
    archive = output / 'iea-stage1-acceptance-v1.tar.gz'
    with tarfile.open(archive, 'w:gz', dereference=True) as tar:
        for path in sorted(files):
            tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
        payload = (json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode()
        info = tarfile.TarInfo('MANIFEST.json')
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    # Verify every archived byte stream against the manifest, not just archive creation.
    with tarfile.open(archive, 'r:gz') as tar:
        for item in entries:
            stream = tar.extractfile(item['path'])
            assert stream is not None
            h = hashlib.sha256()
            total = 0
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                h.update(block)
                total += len(block)
            assert total == item['bytes'] and h.hexdigest() == item['sha256'], item['path']
    manifest_path = output / 'MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    report = dict(archive=str(archive), sha256=digest(archive), bytes=archive.stat().st_size,
                  file_count=len(entries), archived_files_verified=True, manifest=str(manifest_path))
    (output / 'archive-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

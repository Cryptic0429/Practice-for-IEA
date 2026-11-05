"""Freeze the entire current campaign and its reproducibility dependencies.

Archive and progress live outside the frozen project tree. Base shards remain
external with revision and current streaming-hash audit; every dataset image
referenced by the campaign data is included. No activation/credential files.
"""
import datetime
import hashlib
import json
import os
import shutil
import tarfile
import time
from pathlib import Path

ROOT = Path('/mnt/sdc/zhc/IEA')
OUT = ROOT / 'outputs/goal-v2-20261103'
PACKAGE = ROOT / 'artifacts/goal-v2-20261103-packaging'
STAGE = PACKAGE / 'project'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    PACKAGE.mkdir(exist_ok=False)
    STAGE.mkdir()
    state = {'pid': os.getpid(), 'status': 'collecting', 'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    def save():
        state['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        tmp = PACKAGE / 'status.tmp'
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(PACKAGE / 'status.json')
    save()
    try:
        lock = json.loads((OUT / 'final-models-lock.json').read_text())
        for name in lock['models']:
            assert json.loads((OUT / 'final-evaluation' / name / 'status.json').read_text())['status'] == 'completed'
        for name in ['worker-final-0.json', 'worker-final-2.json', 'worker-final-3.json', 'worker-qualityweighted3k.json', 'worker-restore-smoke.json']:
            assert json.loads((OUT / name).read_text())['status'] == 'completed', name
        assert json.loads((OUT / 'final-visual-review-summary.json').read_text())['outputs'] == 210
        assert all(json.loads((OUT / 'delivery-inference-replay-v1/result.json').read_text())['replay_checks'].values())
        assert sha(OUT / 'final-models-lock.json') == json.loads((OUT / 'final-analysis/summary.json').read_text())['model_lock_sha256']
        sources = {}
        def add(path):
            path = Path(path)
            assert path.is_file(), path
            relative = str(path.relative_to(ROOT))
            assert relative not in ['MANIFEST.json']
            sources[relative] = path
        def tree(directory, predicate=lambda p: True):
            for p in directory.rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and predicate(p):
                    add(p)
        tree(OUT)
        tree(ROOT / 'scripts', lambda p: p.suffix == '.py')
        tree(ROOT / 'configs', lambda p: p.suffix in ['.json', '.yaml'] or p.name in ['stage2-rl-requirements.txt', 'constraints.txt', 'repository-commit.txt'])
        tree(ROOT / 'repo/src', lambda p: p.suffix == '.py')
        tree(ROOT / 'repo/LLaMA-Factory/src', lambda p: p.suffix == '.py')
        data = ROOT / 'data/goal-v2-20261103'
        tree(data)
        for relative in ['data/stage1/splits', 'data/stage2/splits']:
            directory = ROOT / relative
            if directory.exists():
                tree(directory, lambda p: p.suffix == '.json')
        assets = set()
        def images(value):
            if isinstance(value, dict):
                for item in value.values():
                    images(item)
            elif isinstance(value, list):
                for item in value:
                    images(item)
            elif isinstance(value, str) and value.startswith(str(ROOT) + '/') and Path(value).suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                p = Path(value)
                if p.is_file():
                    assets.add(p)
        for p in data.rglob('*.json'):
            images(json.loads(p.read_text()))
        for p in assets:
            add(p)
        for name in lock['baseline_keys']:
            for filename in ['adapter_model.safetensors', 'adapter_config.json']:
                add(Path(lock['models'][name]['path']) / filename)
        expanded = ROOT / 'models/goal-v2-A-expanded-start-v1'
        if expanded.exists():
            tree(expanded)
        base = ROOT / 'models/Qwen2.5-VL-7B-Instruct'
        tree(base, lambda p: p.suffix != '.safetensors' and '.cache' not in p.parts)
        for name in ['README.md', '实验记录与结论.md', '原理与复用指南.md']:
            add(ROOT / name)
        tree(ROOT / 'metadata/goal-v2-20261103')
        tree(ROOT / 'artifacts/goal-v2-20261103')
        entries = []
        total_bytes = sum(p.stat().st_size for p in sources.values())
        state.update(status='freezing_and_hashing', files_total=len(sources), bytes_total=total_bytes,
                     referenced_image_assets=len(assets), files_done=0, bytes_done=0)
        save()
        last = time.monotonic()
        for relative, source in sorted(sources.items()):
            target = STAGE / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # Immutable completed artifacts can be hardlinked. SHA verification of
            # every tar member later detects any accidental source mutation.
            os.link(source, target)
            entries.append({'path': relative, 'bytes': target.stat().st_size, 'sha256': sha(target)})
            state['files_done'] += 1
            state['bytes_done'] += target.stat().st_size
            if time.monotonic()-last > 5:
                save()
                last = time.monotonic()
        manifest = {'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'scope': 'All campaign results including negative trials, checkpoints, predictions/images, frozen code/config/environment and referenced dataset assets.',
                    'files': entries, 'file_count': len(entries), 'bytes': total_bytes,
                    'external_dependency': {'model': 'Qwen/Qwen2.5-VL-7B-Instruct', 'revision': 'cc594898137f460bfe9f0759e9844b3ce807cfb5',
                        'weight_shards_included': False, 'audit': 'outputs/goal-v2-20261103/base-source-current-audit.json',
                        'instruction': 'Restore external verified base shards at models/Qwen2.5-VL-7B-Instruct; bundled tokenizer/config files remain exact.'},
                    'excluded': ['venv binaries', 'base safetensor shards', 'activation and credential scripts', 'archive/progress files outside this snapshot'],
                    'final_receipt': 'An external final-acceptance.json records archive SHA and local per-member verification after packaging; no self-referential receipt is embedded.'}
        (STAGE / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        archive = PACKAGE / 'iea-goal-v2-20261103.tar'
        h = hashlib.sha256()
        state.update(status='writing_archive', bytes_written=0)
        save()
        class Writer:
            def __init__(self, file):
                self.file = file
                self.last = time.monotonic()
            def write(self, content):
                count = self.file.write(content)
                h.update(content[:count])
                state['bytes_written'] += count
                if time.monotonic()-self.last > 5:
                    save()
                    self.last = time.monotonic()
                return count
        with archive.open('xb') as file:
            writer = Writer(file)
            with tarfile.open(fileobj=writer, mode='w|', format=tarfile.PAX_FORMAT, dereference=True) as tar:
                # Manifest first permits a one-pass streaming verification.
                tar.add(STAGE / 'MANIFEST.json', arcname='MANIFEST.json', recursive=False)
                for row in entries:
                    tar.add(STAGE / row['path'], arcname=row['path'], recursive=False)
        receipt = {'status': 'archive_written_local_verification_pending', 'archive': str(archive),
                   'archive_bytes': archive.stat().st_size, 'archive_sha256': h.hexdigest(),
                   'manifest_sha256': sha(STAGE / 'MANIFEST.json'), 'files': len(entries),
                   'referenced_image_assets': len(assets), 'model_lock_sha256': sha(OUT / 'final-models-lock.json'),
                   'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        (PACKAGE / 'archive-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        state.update(status='completed', receipt=receipt)
        save()
        print(json.dumps(receipt))
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        save()
        raise

if __name__ == '__main__':
    main()

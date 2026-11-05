"""Hash actual base and released-model shards before final model locking."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/sdc/zhc/IEA')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    output = ROOT / 'configs/evaluation-weight-integrity-v1.json'
    assert not output.exists(), 'Preserve the existing full-hash record.'
    original = json.loads((ROOT / 'configs/qwen-base-revision.json').read_text())
    expected = {item['name']:item for item in original['files']}
    directories = {'Base': ROOT / 'models/Qwen2.5-VL-7B-Instruct',
                   'Official': ROOT / 'models/Qwen2.5-VL-7B-IEA-runtime'}
    models = {}
    for name, directory in directories.items():
        index = json.loads((directory / 'model.safetensors.index.json').read_text())
        filenames = sorted(set(index['weight_map'].values()))
        assert set(filenames) == {path.name for path in directory.glob('*.safetensors')}
        shards = []
        for filename in filenames:
            path = directory / filename
            before = path.stat()
            hashed = digest(path)
            after = path.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            if name == 'Base':
                assert hashed == expected[filename]['sha256'] == expected[filename]['upstream_lfs_sha256']
            shards.append(dict(path=str(path), resolved_path=str(path.resolve()), bytes=after.st_size,
                               mtime_ns=after.st_mtime_ns, sha256=hashed))
            print(name, filename, 'HASH_VERIFIED', flush=True)
        models[name] = dict(directory=str(directory), shards=shards,
                            total_bytes=sum(item['bytes'] for item in shards),
                            indexed_tensors=len(index['weight_map']),
                            upstream_comparison='Matches the pinned original Qwen upstream LFS hashes' if name == 'Base'
                            else 'Content identity of the existing released-model files; not an independent publisher checksum attestation')
    report = dict(verified_at=datetime.now(timezone.utc).isoformat(), models=models,
                  base_revision=original['revision'], purpose='Exact weight identity before final test model locking; no model execution or weight changes.')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('ALL_WEIGHT_CONTENT_HASHES_RECORDED', output, flush=True)


if __name__ == '__main__':
    main()

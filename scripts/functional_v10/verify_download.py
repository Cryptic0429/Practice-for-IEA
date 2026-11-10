"""Independently verify the downloaded frozen archive, including hard links."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import tarfile


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    folder = root / 'archives/functional-v10-20261106'
    receipt = json.loads((folder / 'receipt.json').read_text())
    archive = root / receipt['archive']
    assert archive.stat().st_size == receipt['bytes']
    assert digest_file(archive) == receipt['sha256']
    prefix = 'functional-v10-20261106/'
    with tarfile.open(archive, 'r:gz') as tar:
        data = tar.extractfile(prefix + 'MANIFEST.json').read()
    assert hashlib.sha256(data).hexdigest() == receipt['manifest_sha256']
    manifest = json.loads(data)
    expected = {prefix + row['path']: row for row in manifest['files']}
    stored, verified, hardlinks = {}, set(), 0
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            if member.name not in expected:
                continue
            if member.islnk():
                digest, size = stored[member.linkname]
                hardlinks += 1
            else:
                stream = tar.extractfile(member)
                digest = hashlib.sha256(); size = 0
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk); digest.update(chunk)
                digest = digest.hexdigest()
            row = expected[member.name]
            assert digest == row['sha256'] and size == row['bytes'], member.name
            stored[member.name] = digest, size
            verified.add(member.name)
    assert len(verified) == receipt['verified_files'] == len(expected)
    result = dict(at=dt.datetime.now(dt.timezone.utc).isoformat(), status='local_download_all_files_verified',
                  archive=str(archive), bytes=archive.stat().st_size, sha256=receipt['sha256'],
                  manifest_sha256=receipt['manifest_sha256'], verified_files=len(verified),
                  verified_hardlinks=hardlinks, dependencies=manifest['dependencies'],
                  snapshot_scope=receipt['scope'], round10_included=True)
    (folder / 'download-receipt.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()

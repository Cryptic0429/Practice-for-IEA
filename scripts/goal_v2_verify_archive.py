"""One-pass archive SHA256 plus every regular member SHA256 and coverage."""
import argparse
import datetime
import hashlib
import json
import os
import tarfile
import time
from pathlib import Path, PurePosixPath

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--receipt', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    assert not a.output.exists()
    expected = json.loads(a.receipt.read_text())
    assert a.archive.stat().st_size == expected['archive_bytes']
    status_path = a.output.with_name('archive-verification-status.json')
    state = {'pid': os.getpid(), 'status': 'verifying', 'archive_bytes': a.archive.stat().st_size,
             'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'members_verified': 0, 'member_bytes_verified': 0, 'archive_bytes_read': 0}
    def save():
        state['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        tmp = status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(status_path)
    save()
    archive_hash = hashlib.sha256()
    class Reader:
        def __init__(self, file):
            self.file = file
        def read(self, count=-1):
            content = self.file.read(count)
            archive_hash.update(content)
            state['archive_bytes_read'] += len(content)
            return content
    seen = set()
    last = time.monotonic()
    try:
        with a.archive.open('rb') as file:
            reader = Reader(file)
            with tarfile.open(fileobj=reader, mode='r|') as tar:
                first = tar.next()
                assert first.name == 'MANIFEST.json' and first.isfile()
                data = tar.extractfile(first).read()
                assert hashlib.sha256(data).hexdigest() == expected['manifest_sha256']
                manifest = json.loads(data)
                entries = {row['path']: row for row in manifest['files']}
                assert len(entries) == manifest['file_count'] == expected['files']
                # Iteration includes cached first member; skip that exact header.
                for member in tar:
                    if member.name == 'MANIFEST.json':
                        continue
                    path = PurePosixPath(member.name)
                    assert not path.is_absolute() and '..' not in path.parts
                    assert member.isfile() and member.name in entries and member.name not in seen
                    row = entries[member.name]
                    assert member.size == row['bytes']
                    h = hashlib.sha256()
                    with tar.extractfile(member) as content:
                        for chunk in iter(lambda: content.read(4*1024*1024), b''):
                            h.update(chunk)
                    assert h.hexdigest() == row['sha256'], member.name
                    seen.add(member.name)
                    state['members_verified'] += 1
                    state['member_bytes_verified'] += member.size
                    if time.monotonic()-last > 5:
                        save()
                        last = time.monotonic()
            while reader.read(4*1024*1024):
                pass
        assert seen == set(entries)
        assert state['member_bytes_verified'] == manifest['bytes']
        assert archive_hash.hexdigest() == expected['archive_sha256']
        assert state['archive_bytes_read'] == a.archive.stat().st_size
        receipt = {'passed': True, 'archive': str(a.archive.resolve()),
                   'archive_sha256': archive_hash.hexdigest(), 'archive_bytes': a.archive.stat().st_size,
                   'manifest_sha256': expected['manifest_sha256'], 'verified_members': len(seen),
                   'verified_member_bytes': state['member_bytes_verified'],
                   'referenced_image_assets': expected['referenced_image_assets'],
                   'verification': 'Entire raw tar SHA plus exact member set, member sizes and every member SHA verified locally; no file extraction or unpickling required.',
                   'completed_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        a.output.write_text(json.dumps(receipt, indent=2) + '\n')
        state.update(status='completed', result=receipt)
        save()
        print(json.dumps(receipt))
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        save()
        raise

if __name__ == '__main__':
    main()

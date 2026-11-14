#!/usr/bin/env python3
"""Verify the compact checkout or restore archived evidence without running models."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts or '\n' in value:
        raise ValueError('Expected a relative archive path: ' + value)
    return path.as_posix()


def selected_entries(manifest, prefixes=(), paths=()):
    prefixes = [safe_path(p).rstrip('/') for p in prefixes]
    wanted = set(paths)
    result = [e for e in manifest['entries'] if e['path'] in wanted or any(
        e['path'] == p or e['path'].startswith(p + '/') for p in prefixes)]
    if not result:
        raise ValueError('No matching archive entries')
    for entry in result:
        safe_path(entry['path'])
    return result


def verify(root=ROOT):
    manifest = read(root / 'models/mainline.json')
    errors = []
    for model in manifest['models']:
        for item in model['files']:
            path = root / item['path']
            if not path.is_file() or sha(path) != item['sha256']:
                errors.append(item['path'])
    # Frozen protocol hashes are meaningful only if all required code is retained.
    protocols = ['experiments/functional-v15-20261106/training-protocol.json',
                 'experiments/functional-v21-20261107/training-protocol.json']
    checks = 0
    for name in protocols:
        for name, digest in read(root / name)['code_hashes'].items():
            checks += 1
            if not (root / name).is_file() or sha(root / name) != digest:
                errors.append(name)
    archived = read(root / 'docs/archive/manifest-before.json')
    old_report_checks = 0
    for entry in archived['entries']:
        if (entry['path'].startswith('reports/model-review-20261112/') or
                entry['path'] == 'output/pdf/iea-model-review-20261112.pdf'):
            old_report_checks += 1
            path = root / entry['path']
            if not path.is_file() or sha(path) != entry['sha256']:
                errors.append(entry['path'])
    evidence = read(root / 'reports/mainline-technical-20261113/generated/evidence.json')['sources']
    for name, digest in evidence.items():
        if not (root / name).is_file() or sha(root / name) != digest:
            errors.append(name)
    result = dict(ok=not errors, retained_components=len(manifest['models']),
                  frozen_code_checks=checks, old_report_checks=old_report_checks,
                  retained_evidence_checks=len(evidence), failures=sorted(set(errors)),
                  base_model='SJTU1 only; no GPU inference performed')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return not errors


def restore(args):
    receipt = read(ROOT / 'docs/archive/archive-receipt.json')
    manifest_path = ROOT / 'docs/archive/manifest-before.json'
    if sha(manifest_path) != receipt['manifest_sha256']:
        raise ValueError('Archive manifest checksum mismatch')
    manifest = read(manifest_path)
    paths = []
    if args.profile:
        paths = read(ROOT / 'configs/mainline/restore-profiles.json')[args.profile]['paths']
    entries = selected_entries(manifest, args.prefix, paths)
    dest = Path(args.destination).expanduser().absolute()
    print(json.dumps(dict(destination=str(dest), entries=len(entries),
                         bytes=sum(e.get('bytes', 0) for e in entries),
                         source=receipt['host'] + ':' + receipt['tree'],
                         execute=args.execute), ensure_ascii=False, indent=2))
    if not args.execute:
        print('Preview only. Add --execute to download; no training or inference will run.')
        return
    # Never restore through a symlink or replace different existing content.
    for entry in entries:
        target = dest / entry['path']
        for ancestor in (target, *target.parents):
            if ancestor.is_symlink():
                raise ValueError('Refusing symlink destination: ' + str(ancestor))
            if ancestor == dest:
                break
        if target.exists():
            if entry['type'] != 'file' or not target.is_file() or sha(target) != entry['sha256']:
                raise FileExistsError('Existing content differs: ' + str(target))
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.iea-restore-', dir=dest) as temporary:
        staging = Path(temporary)
        names = staging / 'files.txt'
        names.write_text(''.join(e['path'] + '\n' for e in entries))
        tree = staging / 'tree'
        tree.mkdir()
        subprocess.run(['rsync', '-a', '--files-from=' + str(names),
                        receipt['host'] + ':' + receipt['tree'] + '/', str(tree) + '/'], check=True)
        for entry in entries:
            source = tree / entry['path']
            if entry['type'] == 'file':
                if not source.is_file() or source.is_symlink() or sha(source) != entry['sha256']:
                    raise ValueError('Downloaded content failed verification: ' + entry['path'])
            elif not source.is_symlink() or os.readlink(source) != entry['target']:
                raise ValueError('Downloaded link failed verification: ' + entry['path'])
        for entry in entries:
            source, target = tree / entry['path'], dest / entry['path']
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if entry['type'] == 'symlink':
                # Preserve archive link identity; historical absolute aliases may
                # remain machine-specific. File-based profiles are preferred.
                target.symlink_to(entry['target'])
            else:
                shutil.move(str(source), str(target))
    print('Restored and SHA256-verified. Historical absolute paths may require IEA_ROOT or SJTU1.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('verify')
    restore_parser = sub.add_parser('restore')
    restore_parser.add_argument('--prefix', action='append', default=[])
    restore_parser.add_argument('--profile')
    restore_parser.add_argument('--destination', required=True)
    restore_parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.command == 'verify':
        raise SystemExit(0 if verify() else 1)
    restore(args)


if __name__ == '__main__':
    main()

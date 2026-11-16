#!/usr/bin/env python3
"""Verify model registry layout, frozen protocol hashes, and retained evidence integrity."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BINARY_SUFFIXES = ('.safetensors', '.pt', '.pth', '.bin')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root=ROOT, require_weights=False):
    manifest = read(root / 'models/mainline.json')
    errors = []
    weights_total = 0
    weights_verified = 0

    for model in manifest['models']:
        for alias in model.get('alias_paths', []):
            target = root / alias
            if not target.is_dir() or target.resolve() != (root / model['path']).resolve():
                errors.append(alias)
        for item in model['files']:
            path = root / item['path']
            is_weight = path.suffix in BINARY_SUFFIXES
            if is_weight:
                weights_total += 1
                if not path.is_file():
                    if require_weights:
                        errors.append(item['path'])
                    continue
                if sha(path) != item['sha256']:
                    errors.append(item['path'])
                else:
                    weights_verified += 1
            else:
                if not path.is_file() or sha(path) != item['sha256']:
                    errors.append(item['path'])

    # Frozen protocol hashes are meaningful only if all required code is retained.
    protocols = [
        'experiments/functional-v15-20261106/training-protocol.json',
        'experiments/functional-v21-20261107/training-protocol.json',
    ]
    checks = 0
    for protocol_path in protocols:
        for name, digest in read(root / protocol_path)['code_hashes'].items():
            checks += 1
            if not (root / name).is_file() or sha(root / name) != digest:
                errors.append(name)

    evidence = read(root / 'reports/mainline-technical-20261113/generated/evidence.json')['sources']
    for name, digest in evidence.items():
        if not (root / name).is_file() or sha(root / name) != digest:
            errors.append(name)

    result = dict(
        ok=not errors,
        retained_components=len(manifest['models']),
        weights_verified=f'{weights_verified}/{weights_total}',
        require_weights=require_weights,
        frozen_code_checks=checks,
        retained_evidence_checks=len(evidence),
        failures=sorted(set(errors)),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return not errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    verify_parser = sub.add_parser('verify')
    verify_parser.add_argument(
        '--require-weights',
        action='store_true',
        help='Fail if local binary weights (.safetensors/.pt) are missing.',
    )
    args = parser.parse_args()
    if args.command == 'verify':
        raise SystemExit(0 if verify(require_weights=args.require_weights) else 1)


if __name__ == '__main__':
    main()

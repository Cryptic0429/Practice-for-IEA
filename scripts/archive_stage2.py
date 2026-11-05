"""Package completed Stage 2 evidence; verify every member without extracting it.

Base weights and the original image corpus remain external. All Stage 2 outputs,
including optimizer states and predicted images, are included. No GPU work.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

RUN = 'stage2-A-edit-grpo-v1'
DOCS = ['README.md', 'Stage2_GRPO_方案与执行说明.md',
        'IEA_SJTU1_两天任务表与规划.md', '验收_Stage2_完整实验报告.md']
REQUIRED_AUDITS = {'frozen_inputs', 'grpo_training', 'validation_selection',
                  'sft_control', 'checkpoint_states', 'independent_reloads',
                  'model_lock_before_tests', 'all_900_predictions',
                  'paired_statistics', 'anonymous_90_observations',
                  'figure_direct_and_automated_qa', 'budget', 'chinese_report'}


def sha(path):
    with Path(path).open('rb') as stream:
        return stream_sha(stream)[0]


def stream_sha(stream):
    h = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        h.update(block)
        size += len(block)
    return h.hexdigest(), size


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Unsafe archive path: {name}')
    for part in path.parts:
        lower = part.lower()
        if (lower in {'activate.sh', '.env', '.ssh', '.git', '__pycache__',
                      'credentials', 'credentials.json', 'token.json'}
                or lower.startswith('.env.') or lower.endswith(('.pem', '.key'))):
            raise ValueError(f'Excluded archive path: {name}')
    return path


def verify(archive):
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or names.count('MANIFEST.json') != 1:
            raise ValueError('Duplicate members or missing manifest')
        for member in members:
            safe_name(member.name)
            if not member.isfile():
                raise ValueError(f'Non-regular archive member: {member.name}')
        manifest = json.load(tar.extractfile('MANIFEST.json'))
        entries = manifest['files']
        expected = {e['path'] for e in entries}
        if len(expected) != len(entries) or expected | {'MANIFEST.json'} != set(names):
            raise ValueError('Manifest does not exactly cover archive members')
        for entry in entries:
            digest, size = stream_sha(tar.extractfile(entry['path']))
            if digest != entry['sha256'] or size != entry['bytes']:
                raise ValueError(f'Archive content mismatch: {entry["path"]}')
    return dict(archive=str(archive), sha256=sha(archive), bytes=archive.stat().st_size,
                file_count=len(entries), all_member_hashes_verified=True,
                verified_at=datetime.now(timezone.utc).isoformat())


def collect(root):
    root = root.resolve()
    files = set()

    def add(path):
        relative = path.relative_to(root).as_posix()
        safe_name(relative)
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'Symlink or external file: {relative}')
        if not path.is_file():
            raise ValueError(f'Missing required archive input: {relative}')
        files.add(path)

    for name in DOCS:
        add(root / name)
    for prefix in [f'outputs/{RUN}', f'logs/{RUN}', f'metadata/{RUN}',
                   f'data/stage2/{RUN}']:
        folder = root / prefix
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError(f'Missing or linked directory: {prefix}')
        for path in folder.rglob('*'):
            if path.is_symlink():
                raise ValueError(f'Symlink inside evidence: {path}')
            if path.is_file():
                add(path)
    for path in (root / 'configs').glob('stage2-*'):
        add(path)
    for name in ['evaluation-protocol-v1.json', 'qwen-base-revision.json',
                 'report-installed-packages.txt', 'repository-state.json']:
        add(root / 'configs' / name)
    for path in (root / 'scripts').glob('*.py'):
        add(path)
    for name in ['image_editor.py', 'config.py', 'utils.py']:
        add(root / 'repo/src' / name)
    for name in ['adapter_model.safetensors', 'adapter_config.json']:
        add(root / 'outputs/stage1-A-image-edit-3k-v1' / name)
    return sorted(files)


def build(root, destination):
    root = root.resolve()
    out = root / 'outputs' / RUN
    acceptance_path = out / 'prearchive-acceptance.json'
    acceptance = json.loads(acceptance_path.read_text())
    checks = acceptance['requirements']
    if acceptance['status'] != 'ready_to_archive' or set(checks) != REQUIRED_AUDITS:
        raise ValueError('Complete prearchive requirement audit is required')
    for name, check in checks.items():
        if check.get('passed') is not True or not check.get('evidence'):
            raise ValueError(f'Unverified requirement: {name}')
        for evidence, digest in check['evidence'].items():
            safe_name(evidence)
            if sha(root / evidence) != digest:
                raise ValueError(f'Changed audit evidence: {evidence}')
    pipeline = json.loads((root / 'logs' / RUN / 'experiment-pipeline.json').read_text())
    if pipeline['status'] != 'numerical_experiment_complete_pending_visual_review_and_report':
        raise ValueError('Numerical pipeline has not completed')
    files = collect(root)
    # Keep the archive outside the recursively collected evidence directories.
    for parent in ['outputs/' + RUN, 'logs/' + RUN, 'metadata/' + RUN, 'data/stage2/' + RUN]:
        if destination.resolve().is_relative_to(root / parent):
            raise ValueError('Archive destination overlaps evidence')
    destination.mkdir(parents=True, exist_ok=False)
    entries = [dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size,
                    sha256=sha(p)) for p in files]
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), files=entries,
                    scope='All Stage2 outputs, recoverable checkpoints, predictions/images, '
                          'candidate logs, reports, frozen inputs/configs/environment and source.',
                    external_assets={'base_weights': str(root / 'models/Qwen2.5-VL-7B-Instruct'),
                                     'original_image_corpus': str(root / 'repo/datasets')},
                    exclusions='Activation script, credentials, installed environments and base weights.',
                    prearchive_audit_sha256=sha(acceptance_path))
    payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    archive = destination / 'iea-stage2-acceptance-v1.tar.gz'
    with tarfile.open(archive, 'x:gz', dereference=False) as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
        info = tarfile.TarInfo('MANIFEST.json')
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    report = verify(archive)
    (destination / 'MANIFEST.json').write_bytes(payload)
    (destination / 'archive-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['build', 'verify'])
    parser.add_argument('--root', type=Path, default=Path('/mnt/sdc/zhc/IEA'))
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.mode == 'build':
        if args.destination is None:
            parser.error('build requires --destination')
        report = build(args.root.resolve(), args.destination)
    else:
        if args.archive is None:
            parser.error('verify requires --archive')
        report = verify(args.archive)
    if args.report:
        with args.report.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

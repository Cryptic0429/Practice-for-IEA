"""Organize delivered IEA files while retaining legacy runtime paths.

Uses same-filesystem renames, records every move, and leaves absolute compatibility
links where historical scripts and receipts still reference an old location.
No model, dataset, experiment result, or frozen source contents are changed.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def plan(root, mode):
    operations = []

    def move(old, new, alias=True):
        src, dst = root / old, root / new
        if not src.exists() and not src.is_symlink():
            return
        if src.is_symlink() and src.resolve() == dst.resolve():
            return
        if dst.exists() or dst.is_symlink():
            raise FileExistsError(str(dst))
        operations.append({'old': old, 'new': new, 'keep_legacy_link': alias})

    if mode == 'local':
        move('.downloads', 'data/sources')
        move('IEA-CVPR2026Findings.pdf', 'docs/papers/IEA-CVPR2026Findings.pdf', False)
        for name in ['.gier-selective.log', '.stage1-download.log', '.stage1-selective.log']:
            move(name, 'archives/download-logs/' + name.lstrip('.'), False)
        move('artifacts/adapters', 'models/stage1')
        move('artifacts/stage2-acceptance-v1/models', 'models/stage2')
        move('artifacts/goal-v2-20261103/adapters', 'models/quality-v2-20261103')
        for old, stage, archive in [
            ('artifacts/acceptance-v1', 'stage1', 'iea-stage1-acceptance-v1.tar.gz'),
            ('artifacts/stage2-acceptance-v1', 'stage2', 'iea-stage2-acceptance-v1.tar.gz'),
            ('artifacts/goal-v2-20261103', 'quality-v2-20261103', 'iea-goal-v2-20261103.tar')]:
            move(old + '/' + archive, 'archives/' + stage + '/' + archive)
        move('artifacts/goal-v2-20261103/blind-review', 'experiments/quality-v2-20261103/review')
        move('artifacts/stage2-review', 'experiments/stage2/review')
        move('artifacts/acceptance-v1', 'experiments/stage1/delivery')
        move('artifacts/stage2-acceptance-v1', 'experiments/stage2/delivery')
        move('artifacts/goal-v2-20261103', 'experiments/quality-v2-20261103/delivery')
        move('artifacts/analysis-20261103', 'experiments/diagnostics/20261103')
        move('artifacts/diagnostics', 'experiments/diagnostics/20261104')
        move('artifacts/document-history', 'archives/document-history')
    else:
        move('archives/stage2-acceptance-v1', 'archives/stage2')
        move('outputs/stage1-A-image-edit-3k-v1', 'experiments/stage1/runs/A')
        move('outputs/stage1-B-image-edit-3k-v1', 'experiments/stage1/runs/B')
        move('outputs/evaluation-v1', 'experiments/stage1/evaluation')
        move('outputs/acceptance-v1/iea-stage1-acceptance-v1.tar.gz',
             'archives/stage1/iea-stage1-acceptance-v1.tar.gz')
        move('outputs/acceptance-v1', 'experiments/stage1/delivery')
        move('outputs/stage2-A-edit-grpo-v1', 'experiments/stage2/run')
        move('outputs/goal-v2-20261103', 'experiments/quality-v2-20261103/runs')
        move('artifacts/goal-v2-20261103/adapters', 'models/quality-v2-20261103')
        move('artifacts/goal-v2-20261103/blind-review', 'experiments/quality-v2-20261103/review')
        move('artifacts/goal-v2-20261103', 'experiments/quality-v2-20261103/delivery')
        move('artifacts/goal-v2-20261103-packaging/iea-goal-v2-20261103.tar',
             'archives/quality-v2-20261103/iea-goal-v2-20261103.tar')
        move('artifacts/goal-v2-20261103-packaging',
             'archives/quality-v2-20261103/packaging')
        move('artifacts/goal-v2-20261103-packaging-attempt1-failed',
             'archives/quality-v2-20261103/failed-packaging-attempt1')
        main_docs = {'README.md', '实验记录与结论.md', '原理与复用指南.md'}
        for p in sorted(root.glob('*.md')):
            if p.name not in main_docs:
                move(p.name, 'docs/history/' + p.name, False)
    for old, new in [
        ('metadata/evaluation-v1', 'experiments/stage1/evaluation-evidence'),
        ('metadata/image-edit-3k-v1', 'experiments/stage1/data-audit'),
        ('metadata/stage1-A-3k-run', 'experiments/stage1/training-A'),
        ('metadata/stage1-B-3k-run', 'experiments/stage1/training-B'),
        ('metadata/stage2-A-edit-grpo-v1', 'experiments/stage2/metadata'),
        ('metadata/goal-v2-20261103', 'experiments/quality-v2-20261103/metadata'),
        ('metadata/benchmarks', 'experiments/shared/benchmarks')]:
        move(old, new)
    for p in sorted((root / 'metadata').iterdir()):
        if p.is_file() and not p.is_symlink():
            move('metadata/' + p.name, 'experiments/shared/metadata/' + p.name)
    return operations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=['local', 'server'], required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    operations = plan(root, args.mode)
    if not args.apply:
        print(json.dumps({'root': str(root), 'mode': args.mode,
                          'planned_moves': len(operations), 'operations': operations},
                         ensure_ascii=False, indent=2))
        return
    audit_dir = root / 'archives/organization-20261104'
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_dir / (args.mode + '-moves.json')
    if audit.exists():
        report = json.loads(audit.read_text())
        if report['status'] != 'in_progress':
            raise FileExistsError('An organization receipt already exists: ' + str(audit))
    else:
        report = {'root': str(root), 'mode': args.mode, 'status': 'in_progress',
                  'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'operations': [], 'policy': 'Content-preserving renames; old runtime paths retained as links.'}

    def save():
        tmp = audit.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(audit)

    save()
    for op in operations:
        src, dst = root / op['old'], root / op['new']
        old_stat = src.stat()
        is_file = src.is_file()
        digest = sha(src) if is_file and old_stat.st_size < 8 * 1024 * 1024 else None
        # Remember relative links, whose parent depth changes during a rename.
        relative_links = []
        if src.is_dir():
            for dirname, dirs, files in os.walk(src, followlinks=False):
                for name in dirs + files:
                    link = Path(dirname) / name
                    if link.is_symlink() and not os.path.isabs(os.readlink(link)):
                        relative_links.append((link.relative_to(src), str(link.resolve())))
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dst)
        assert (dst.stat().st_dev, dst.stat().st_ino) == (old_stat.st_dev, old_stat.st_ino)
        for relative, target in relative_links:
            link = dst / relative
            try:
                new_target = str(dst / Path(target).relative_to(src))
            except ValueError:
                new_target = target
            if link.resolve() != Path(new_target):
                link.unlink()
                link.symlink_to(new_target)
        if op['keep_legacy_link']:
            src.symlink_to(str(dst), target_is_directory=dst.is_dir())
            assert src.resolve() == dst.resolve()
        if digest:
            assert sha(dst) == digest
        report['operations'].append(dict(op, inode_preserved=True, sha256=digest))
        save()
    report.update(status='completed', finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save()
    print(json.dumps({'status': 'completed', 'mode': args.mode, 'moves': len(operations),
                      'receipt': str(audit)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

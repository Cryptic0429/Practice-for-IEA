"""Guard B's sampling-only change and immutable source pool."""
import hashlib
import json
from pathlib import Path
import subprocess
import yaml

ROOT = Path('/mnt/sdc/zhc/IEA')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    subprocess.run([str(ROOT / 'scripts/train_stage1_frozen.sh'), '--check-only'], check=True)
    m = json.loads((ROOT / 'configs/stage1-B-3k-manifest.json').read_text())
    a = yaml.safe_load((ROOT / 'configs/qlora-stage1-A-3k.yaml').read_text())
    b = yaml.safe_load(Path(m['config_path']).read_text())
    assert sha(Path(m['config_path'])) == m['config_sha256']
    assert sha(Path(m['source_train_path'])) == m['source_train_sha256']
    assert sha(Path(m['train_path'])) == m['train_sha256']
    assert set(a) == set(b)
    assert sorted(key for key in a if a[key] != b[key]) == ['dataset', 'output_dir']
    registry = json.loads((ROOT / 'data/dataset_info.json').read_text())
    assert registry[b['dataset']]['file_name'] == m['train_path']
    source = {row['id']: row for row in json.loads(Path(m['source_train_path']).read_text())}
    rows = json.loads(Path(m['train_path']).read_text())
    assert len(rows) == 3000 and len({row['id'] for row in rows}) == 3000
    for row in rows:
        original = source[row['sampling_source_id']]
        assert {k: v for k, v in row.items() if k not in ('id', 'sampling_source_id')} == {k: v for k, v in original.items() if k != 'id'}
    print('B_FROZEN_PREFLIGHT_PASSED: sampling only; original Qwen; 3000 fixed draws; 376 steps')


if __name__ == '__main__':
    main()

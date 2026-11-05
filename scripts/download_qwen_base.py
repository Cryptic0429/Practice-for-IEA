"""Download the pinned original Qwen weights on SJTU1 and verify SHA-256."""
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import hf_transfer
import requests
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

ROOT = Path('/mnt/sdc/zhc/IEA')
MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct'
REVISION = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'
DEST = ROOT / 'models' / 'Qwen2.5-VL-7B-Instruct'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch(item):
    name, expected_size = item.rfilename, item.size
    expected_sha = item.lfs.sha256 if item.lfs else None
    path = DEST / name
    for attempt in range(3):
        try:
            if path.exists() and path.stat().st_size == expected_size:
                actual = digest(path)
                if expected_sha and actual == expected_sha:
                    return dict(name=name, size=expected_size, sha256=actual, verified=True)
            meta = get_hf_file_metadata(hf_hub_url(MODEL, name, revision=REVISION), timeout=30)
            part = path.with_suffix(path.suffix + '.part')
            print(f'START {name} bytes={expected_size} attempt={attempt+1}', flush=True)
            if expected_sha:
                hf_transfer.download(url=meta.location, filename=str(part), max_files=8,
                                     chunk_size=8*1024*1024, parallel_failures=8, max_retries=5)
            else:
                response = requests.get(meta.location, timeout=(30, 120))
                response.raise_for_status()
                part.write_bytes(response.content)
            actual = digest(part)
            if part.stat().st_size != expected_size or (expected_sha and actual != expected_sha):
                raise ValueError(f'Integrity mismatch: {name}')
            part.replace(path)
            print(f'VERIFIED {name} sha256={actual}', flush=True)
            return dict(name=name, size=expected_size, sha256=actual,
                        verified=True, upstream_lfs_sha256=expected_sha)
        except Exception as e:
            print(f'RETRY {name}: {type(e).__name__}: {str(e)[:160]}', flush=True)
            if attempt == 2:
                raise
            time.sleep(3)


if __name__ == '__main__':
    DEST.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True, timeout=30)
    selected = [s for s in info.siblings if s.rfilename.endswith(('.json', '.txt', '.safetensors', '.md'))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(fetch, selected))
    manifest = dict(model_id=MODEL, revision=REVISION, local_path=str(DEST), files=files)
    (ROOT/'configs'/'qwen-base-revision.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('QWEN_BASE_DOWNLOAD_COMPLETE', flush=True)

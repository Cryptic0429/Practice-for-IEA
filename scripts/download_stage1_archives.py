"""Download the author's four split-archive volumes using Mac connectivity.

Only public Google Drive URLs are used. Downloads resume with validated ranges.
Copies are transferred to SJTU1 after size/SHA-256 checks.
"""
import concurrent.futures
import hashlib
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / '.downloads' / 'iea-release'
REMOTE = '/mnt/sdc/zhc/IEA/data/releases/iea-v1.0-20260606'
STEM = 'Image_Edit_Agent_data_release_v1.0_20260606'
FILES = [
    ('11cZ2qV7IVgoz7wHZS1XyLJAoX-Mm4ybq', '.z01', 4294967296),
    ('1m01Vv1fbjYoY43tBvyXv1g-MdLix9i_h', '.z02', 4294967296),
    ('1xr0OQUZMDevwBihUpvWZrP7Q29teOj32', '.z03', 4294967296),
    ('1xiU6LSZvs328zn8Xsa-RUQ5etwgI7PBE', '.zip', 2858323030),
]


class Form(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.params = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'form' and attrs.get('id') == 'download-form':
            self.action = attrs.get('action')
        if tag == 'input' and attrs.get('type') == 'hidden':
            self.params[attrs['name']] = attrs['value']


def url_for(file_id):
    url = 'https://drive.google.com/uc?' + urllib.parse.urlencode(dict(export='download', id=file_id))
    with urllib.request.urlopen(url, timeout=60) as response:
        if 'text/html' not in response.headers.get('Content-Type', ''):
            return response.url
        parser = Form()
        parser.feed(response.read().decode())
    if not parser.action or urllib.parse.urlsplit(parser.action).hostname != 'drive.usercontent.google.com':
        raise RuntimeError('No public download confirmation form')
    return parser.action + '?' + urllib.parse.urlencode(parser.params)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(item):
    file_id, suffix, size = item
    name = STEM + suffix
    path = DEST / name
    part = path.with_name(name + '.part')
    for attempt in range(5):
        try:
            if path.exists() and path.stat().st_size == size:
                break
            offset = part.stat().st_size if part.exists() else 0
            if offset < size:
                req = urllib.request.Request(url_for(file_id), headers={'Range': f'bytes={offset}-'})
                with urllib.request.urlopen(req, timeout=120) as response:
                    if 'text/html' in response.headers.get('Content-Type', ''):
                        raise RuntimeError('Download returned HTML')
                    content_range = response.headers.get('Content-Range', '')
                    if response.status == 206:
                        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
                        if not match or int(match[1]) != offset or int(match[3]) != size:
                            raise RuntimeError('Unexpected partial response')
                    elif offset:
                        raise RuntimeError('Server did not honor resume range')
                    print(f'START {suffix} offset={offset} bytes={size}', flush=True)
                    last = time.monotonic()
                    with part.open('ab' if offset else 'wb') as output:
                        for chunk in iter(lambda: response.read(4*1024*1024), b''):
                            output.write(chunk)
                            offset += len(chunk)
                            if time.monotonic()-last > 30:
                                print(f'PROGRESS {suffix} {offset}/{size} ({100*offset/size:.1f}%)', flush=True)
                                last = time.monotonic()
            if part.stat().st_size != size:
                raise RuntimeError('Incomplete archive volume')
            part.replace(path)
            break
        except Exception as e:
            print(f'RETRY {suffix} attempt={attempt+1} {type(e).__name__}: {str(e)[:160]}', flush=True)
            if attempt == 4:
                raise
            time.sleep(5)
    digest = sha256(path)
    print(f'LOCAL_VERIFIED {suffix} sha256={digest}', flush=True)
    subprocess.run(['scp', str(path), f'SJTU1:{REMOTE}/{name}.upload'], check=True)
    remote_command = (f"cd {REMOTE} && echo '{digest}  {name}.upload' | sha256sum -c - "
                      f"&& mv {name}.upload {name}")
    subprocess.run(['ssh', 'SJTU1', remote_command], check=True)
    print(f'REMOTE_VERIFIED {suffix}', flush=True)
    return dict(name=name, google_drive_id=file_id, bytes=size, sha256=digest)


if __name__ == '__main__':
    DEST.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ssh', 'SJTU1', f'mkdir -p {REMOTE}'], check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(fetch, FILES))
    manifest = dict(source='https://drive.google.com/drive/folders/1UhM9DViDZkU6NyFKidIMAnF9cZoAXUha',
                    files=files, checksum_note='Computed locally; verified after SSH transfer; publisher checksum not provided.')
    (DEST/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    subprocess.run(['scp', str(DEST/'manifest.json'), f'SJTU1:{REMOTE}/manifest.json'], check=True)
    print('STAGE1_ARCHIVES_DOWNLOAD_COMPLETE', flush=True)

"""Verified selective extraction from public ZIP files supporting HTTP Range."""
import concurrent.futures
import hashlib
import http.client
import json
import os
import struct
import time
import threading
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

from download_stage1_archives import url_for

_LOCAL = threading.local()


def parse_central(data):
    rows, pos = [], 0
    while pos < len(data):
        h = struct.unpack_from('<4s6H3I5H2I', data, pos)
        if h[0] != b'PK\x01\x02':
            raise ValueError('Invalid central directory signature')
        name = data[pos+46:pos+46+h[10]].decode('utf-8' if h[3]&2048 else 'cp437')
        row = dict(name=name, method=h[4], flags=h[3], crc32=h[7], compressed=h[8],
                   uncompressed=h[9], disk=h[13], offset=h[16])
        if any(row[k] == 0xffffffff for k in ['compressed','uncompressed','offset']):
            raise ValueError('ZIP64 per-file sizes are unsupported')
        rows.append(row)
        pos += 46+h[10]+h[11]+h[12]
    return rows


def read_range(url, start, end, total):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'https':
        raise ValueError('HTTPS source required')
    connection = getattr(_LOCAL, 'connection', None)
    if connection is None or connection.host != parsed.hostname:
        connection = http.client.HTTPSConnection(parsed.hostname, timeout=120)
        _LOCAL.connection = connection
    try:
        connection.request('GET', parsed.path+'?'+parsed.query,
                           headers={'Range': f'bytes={start}-{end-1}'})
        r = connection.getresponse()
        if r.status != 206 or r.getheader('Content-Range') != f'bytes {start}-{end-1}/{total}':
            raise ValueError('Server returned an unexpected HTTP range')
        data = r.read()
    except Exception:
        connection.close()
        _LOCAL.connection = None
        raise
    if len(data) != end-start:
        raise ValueError('Truncated HTTP range')
    return data


def extract_selected(rows, selected, sources, dest, cache, workers=8, merge_gap=32768):
    """sources: list of dict(id, size); rows from parse_central or saved index."""
    dest, cache = Path(dest), Path(cache)
    dest.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    bases, total = [], 0
    for s in sources:
        bases.append(total)
        total += s['size']
    def start(r): return bases[r['disk']] + r['offset']
    ordered = sorted(rows, key=start)
    following = {start(a):start(b) for a,b in zip(ordered,ordered[1:])}
    intervals = []
    for r in sorted(selected,key=start):
        if r['name'].endswith('/'):
            continue
        a = start(r)
        b = following.get(a, a+30+len(r['name'].encode())+65535+r['compressed'])
        b = min(b, total)
        if intervals and a <= intervals[-1][1]+merge_gap:
            intervals[-1][1] = max(intervals[-1][1], b)
        else:
            intervals.append([a,b])
    jobs = []
    for a,b in intervals:
        while a < b:
            disk = max(i for i,base in enumerate(bases) if base <= a)
            end = min(a+8*1024*1024,b,bases[disk]+sources[disk]['size'])
            jobs.append((disk,a,end))
            a = end
    urls = {disk:url_for(sources[disk]['id']) for disk in {job[0] for job in jobs}}
    print(f'REQUIRED files={len(selected)} compressed_ranges={sum(b-a for _,a,b in jobs)} chunks={len(jobs)}', flush=True)
    def fetch(job):
        disk,a,b = job
        file = cache/f'{disk}-{a}-{b}.bin'
        for attempt in range(4):
            try:
                if not file.exists() or file.stat().st_size != b-a:
                    data = read_range(urls[disk],a-bases[disk],b-bases[disk],sources[disk]['size'])
                    part = file.with_suffix('.part')
                    part.write_bytes(data)
                    part.replace(file)
                print(f'CHUNK_OK {a}-{b}', flush=True)
                return job,file
            except Exception as e:
                print(f'RETRY {a}-{b} {type(e).__name__}: {str(e)[:120]}',flush=True)
                if attempt==3: raise
                time.sleep(2)
    sparse = cache/'selected-archive.sparse'
    with sparse.open('w+b') as archive:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for (_,a,b),file in pool.map(fetch,jobs):
                archive.seek(a)
                archive.write(file.read_bytes())
        archive.flush()
        records = []
        for row in selected:
            name = row['name']
            if name.endswith('/'):
                continue
            path = (dest/name).resolve()
            if not path.is_relative_to(dest.resolve()):
                raise ValueError('Unsafe archive path')
            archive.seek(start(row))
            header = archive.read(30)
            h = struct.unpack('<4s5H3I2H',header)
            if h[0] != b'PK\x03\x04' or h[2]&1 or h[3] != row['method']:
                raise ValueError(f'Unsupported local header: {name}')
            localname = archive.read(h[9]).decode('utf-8' if h[2]&2048 else 'cp437')
            if localname != name: raise ValueError('Local/central filename mismatch')
            archive.read(h[10])
            compressed = archive.read(row['compressed'])
            if row['method'] == 8:
                content = zlib.decompress(compressed, -15)
            elif row['method'] == 0:
                content = compressed
            else:
                raise ValueError('Unsupported compression')
            if len(content)!=row['uncompressed'] or zlib.crc32(content)!=row['crc32']:
                raise ValueError(f'ZIP size/CRC mismatch: {name}')
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(content)
            records.append(dict(**row,sha256=hashlib.sha256(content).hexdigest()))
        manifest = dict(sources=sources, files=records, crc_verified=True,
                        download_method='HTTP byte ranges; complete selected entries; unselected entries omitted')
        (dest/'extraction-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'EXTRACTION_COMPLETE {len(records)} files',flush=True)
    return records

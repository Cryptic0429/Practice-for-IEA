"""Keep the upstream file; remove its invalid ICC chunk in a pixel-identical copy."""
import hashlib
import json
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT=Path('/mnt/sdc/zhc/IEA')
source=ROOT/'repo/datasets/GIER/images/bpcglv_67NXn92.png'
dest=ROOT/'data/stage1/compatible-images/bpcglv_67NXn92.png'
data=source.read_bytes()
assert data[:8]==b'\x89PNG\r\n\x1a\n'
parts=[data[:8]]
pos=8
removed=[]
while pos<len(data):
    size=struct.unpack_from('>I',data,pos)[0]
    tag=data[pos+4:pos+8]
    payload=data[pos+8:pos+8+size]
    crc=struct.unpack_from('>I',data,pos+8+size)[0]
    if zlib.crc32(tag+payload)!=crc:
        assert tag==b'iCCP', 'Unexpected pixel/header corruption'
        removed.append(dict(chunk=tag.decode(),bytes=size,reason='invalid CRC'))
    else:
        parts.append(data[pos:pos+size+12])
    pos+=size+12
assert removed==[dict(chunk='iCCP',bytes=387,reason='invalid CRC')]
dest.parent.mkdir(parents=True,exist_ok=True)
dest.write_bytes(b''.join(parts))
before=cv2.imread(str(source),cv2.IMREAD_UNCHANGED)
after=cv2.imread(str(dest),cv2.IMREAD_UNCHANGED)
assert before is not None and np.array_equal(before,after)
with Image.open(dest) as im: im.verify()
with Image.open(dest) as im: im.load()
report={str(source):dict(replacement=str(dest),source_sha256=hashlib.sha256(data).hexdigest(),
                        replacement_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                        removed_metadata=removed,pixels_identical=True,pixel_dtype=str(before.dtype),
                        shape=list(before.shape))}
(ROOT/'data/stage1/image-repairs.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

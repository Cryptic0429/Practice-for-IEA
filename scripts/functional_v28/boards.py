"""Create anonymous ordered visual comparisons from exact source/candidate pixels."""
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageOps

from functional_v1.common import ROOT,read,sha,stamp,write
from functional_v15.editor import load_rgb
from functional_v15.outcomes import execute

EXP = ROOT/'experiments/functional-v28-20261107'
PROTOCOL = EXP/'protocol.json'


def open_bound(artifact):
    path = ROOT/artifact['path']
    if sha(path)!=artifact['sha256']:
        raise ValueError(f'Image source changed: {path}')
    return load_rgb(path)


def board(source,edits,crop):
    height,width = source.shape[:2]
    cw,ch = min(212,width),min(136,height)
    cx,cy = crop
    x = max(0,min(width-cw,round(width*cx-cw/2)))
    y = max(0,min(height-ch,round(height*cy-ch/2)))
    box = (x,y,x+cw,y+ch)
    output = Image.new('RGB',(672,364),'white')
    draw = ImageDraw.Draw(output)
    for i,(role,arr) in enumerate(zip(('ORIGINAL','EDIT A','EDIT B'),(source,*edits))):
        if arr.shape!=source.shape:
            raise ValueError('Candidate not aligned with original')
        a = Image.fromarray(arr)
        tile = ImageOps.contain(a,(214,168))
        ox = i*224
        output.paste(tile,(ox+(224-tile.width)//2,28+(168-tile.height)//2))
        draw.text((ox+5,7),role,fill='black')
        draw.text((ox+5,204),'Matched native-pixel detail',fill='black')
        output.paste(a.crop(box),(ox+6,221))
    return output,box


def main():
    plan = read(PROTOCOL)
    folder = EXP/'boards'
    output_dir = EXP/'new-candidates'
    folder.mkdir(parents=True,exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    rows = []
    for c in plan['cases']:
        cid = c['id']
        source = open_bound(c['source'])
        candidates,images = [],[]
        for i,variant in enumerate(c['candidates'],1):
            if 'parameters' in variant:
                decision = dict(action='edit',parameters=variant['parameters'])
                pixels = execute(source,decision)['result']
                path = output_dir/f'{cid}-{i}.png'
                Image.fromarray(pixels).save(path)
                item = dict(path=str(path.relative_to(ROOT)),sha256=sha(path),decision=decision)
                if not np.array_equal(load_rgb(path),pixels):
                    raise ValueError('Editor result did not round trip')
            else:
                item,pixels = variant,open_bound(variant)
            candidates.append(item)
            images.append(pixels)
        comparison = {}
        for name,order in [('12',[0,1]),('21',[1,0])]:
            rendered,box = board(source,[images[i] for i in order],c['crop'])
            path = folder/f'{cid}-{name}.png'
            rendered.save(path)
            comparison[name] = dict(path=str(path.relative_to(ROOT)),sha256=sha(path),
                                    order=[i+1 for i in order],native_box=box)
        rows.append(dict(id=cid,source=c['source'],candidates=candidates,boards=comparison))
    write(EXP/'boards.json',dict(at=stamp(),protocol_sha256=sha(PROTOCOL),cases=rows,GPU_hours=0))


if __name__ == '__main__':
    main()

"""Render matching original-resolution regions for the new held sources."""
from PIL import Image,ImageDraw

from functional_v1.common import ROOT,read,sha,write
from .prepare import EXP

REGIONS = {
    'p346':('window',.83,.14),
    'p377':('suit',.40,.69),
    'p150':('face',.66,.29),
    'p358':('faces',.49,.34),
    'p357':('sky',.26,.16),
    'p309':('sky',.37,.12),
    'p771':('sky',.48,.13),
    'p476':('foliage',.81,.22),
}


def main():
    held = EXP/'held'
    replay = held/'replay.json'
    regions = {}
    for case in read(replay)['cases']:
        cid = case['id']
        name,cx,cy = REGIONS[cid]
        w,h = case['full_frame_size']
        cw,ch = min(w,276),min(h,188)
        x = max(0,min(w-cw,round(w*cx-cw/2)))
        y = max(0,min(h-ch,round(h*cy-ch/2)))
        box = (x,y,x+cw,y+ch)
        sheet = Image.new('RGB',(1120,221),'white')
        draw = ImageDraw.Draw(sheet)
        for i,a in enumerate(case['candidates']):
            path = ROOT/a['path']
            if sha(path)!=a['sha256']:
                raise ValueError(f'Replay image differs: {cid}/{i}')
            with Image.open(path) as image:
                sheet.paste(image.crop(box),(i*280+2,26))
            title = 'Input' if not i else f'Candidate {i}'
            draw.text((i*280+3,5),f'{cid} | {title} | {name}',fill='black')
        output = held/f'{cid}-{name}.png'
        sheet.save(output)
        regions[cid] = dict(crop=box,path=output.name,sha256=sha(output))
    write(held/'detail.json',dict(replay_sha256=sha(replay),regions=regions))


if __name__ == '__main__':
    main()

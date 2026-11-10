"""Bind extra native crops around the requested held image areas."""
from PIL import Image,ImageDraw

from functional_v1.common import ROOT,read,sha,write
from .cache import EXP

REGIONS = {'p021':(.38,.55),'p457':(.48,.28),'p165':(.46,.77),'p102':(.15,.78)}


def main():
    replay = EXP/'replay.json'
    results = {}
    for c in read(replay)['cases']:
        cid = c['id']
        if cid not in REGIONS:
            continue
        original = ROOT/c['source']
        with Image.open(ROOT/c['results']['shadow_up']['path']) as image:
            w,h = image.size
        cx,cy = REGIONS[cid]
        cw,ch = min(w,212),min(h,152)
        x,y = max(0,min(w-cw,round(w*cx-cw/2))),max(0,min(h-ch,round(h*cy-ch/2)))
        box = (x,y,x+cw,y+ch)
        for scope in ('shadow','global'):
            paths = [original,ROOT/c['results'][scope+'_up']['path'],ROOT/c['results'][scope+'_down']['path']]
            sheet = Image.new('RGB',(672,176),'white')
            draw = ImageDraw.Draw(sheet)
            for i,path in enumerate(paths):
                from PIL import ImageOps
                with Image.open(path) as image:
                    tile = ImageOps.exif_transpose(image).convert('RGB').crop(box)
                    sheet.paste(tile,(i*224+6,22))
                draw.text((i*224+6,4),('ORIGINAL','EDIT A','EDIT B')[i],fill='black')
            output = EXP/'images'/f'{cid}-{scope}-detail.png'
            sheet.save(output)
            results[f'{cid}-{scope}'] = dict(path=str(output.relative_to(ROOT)),sha256=sha(output),crop=box)
    write(EXP/'details.json',dict(replay_sha256=sha(replay),crops=results))


if __name__ == '__main__':
    main()

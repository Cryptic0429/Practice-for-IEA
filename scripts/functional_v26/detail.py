"""Extract supplemental matched native-pixel regions from verified replay outputs."""
from PIL import Image,ImageDraw

from functional_v1.common import ROOT,read,sha,write
from .shortlist import EXP

REGIONS = {
    'p345':('bright_trees',.83,.20), 'p189':('white_fabric',.36,.51),
    'p292':('left_face',.29,.50), 'p325':('two_faces',.35,.44),
    'p129':('sky',.70,.23), 'p456':('stars',.46,.27),
    'p526':('reflection',.58,.66), 'p665':('stars',.44,.31),
    'p278':('rock',.64,.89), 'p278-clouds':('clouds',.27,.24), 'p493':('sun',.44,.40),
}


def main():
    manifest = EXP/'replay.json'
    rows = {}
    cases = {c['id']:c for c in read(manifest)['cases']}
    for key,(name,cx,cy) in REGIONS.items():
        cid = key.split('-')[0]
        c = cases[cid]
        w,h = c['full_frame_size']
        cw,ch = min(w,276),min(h,188)
        x = max(0,min(w-cw,round(w*cx-cw/2)))
        y = max(0,min(h-ch,round(h*cy-ch/2)))
        box = (x,y,x+cw,y+ch)
        sheet = Image.new('RGB',(1120,221),'white')
        draw = ImageDraw.Draw(sheet)
        for i,a in enumerate(c['candidates']):
            path = ROOT/a['path']
            if sha(path) != a['sha256']:
                raise ValueError(f'Replay image mismatch: {cid}/{i}')
            with Image.open(path) as source:
                sheet.paste(source.crop(box),(i*280+2,26))
            label = 'Input' if not i else f'Candidate {i}'
            title = f'{label} | upper cloud detail' if key == 'p278-clouds' else f'{cid} | {label} | {name}'
            draw.text((i*280+3,5),title,fill='black')
        target = EXP/f'{cid}-{name}.png'
        sheet.save(target)
        rows[key] = dict(region='upper_clouds' if key == 'p278-clouds' else name,
                         crop=box,panel=target.name,sha256=sha(target))
    write(EXP/'detail.json',dict(replay_sha256=sha(manifest),regions=rows))


if __name__ == '__main__':
    main()

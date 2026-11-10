"""Render fixed comparison grid and isolate the held photo sources."""
import argparse
from itertools import combinations
from pathlib import Path
import shutil

from PIL import Image,ImageDraw,ImageOps

from functional_v1.common import ROOT,read,sha,stamp,write
from functional_v15.editor import load_rgb,save
from functional_v24.probe import protected_sources,SourceMatcher
from .constants import DATA,EXP,PLAN,RECIPES
from .editor import apply


def isolate(cases):
    protected = protected_sources()
    for ver,name in [(24,'reviewed-candidates'),(26,'reviewed-candidates'),(27,'held-cases')]:
        for r in read(ROOT/f'data/training/functional-v{ver}-20261107/{name}.json'):
            protected[str(ROOT/r['image'])]='previously_reviewed'
    for r in read(ROOT/'experiments/functional-v29-20261107/replay.json')['cases']:
        protected[str(ROOT/r['source'])]='r29_reviewed'
    missing = {p:k for p,k in protected.items() if not Path(p).is_file()}
    if set(missing.values())-{'GIER_val','GIER_test'}:
        raise ValueError('Missing protected internal originals')
    available = {p:k for p,k in protected.items() if p not in missing}
    matcher = SourceMatcher()
    flags = []
    for c in cases:
        for other,kind in available.items():
            match = matcher.compare(ROOT/c['path'],other)
            if match:
                flags.append(dict(id=c['id'],other=other,kind=kind,match=match))
        print('Isolated',c['id'],flush=True)
    for a,b in combinations(cases,2):
        match = matcher.compare(ROOT/a['path'],ROOT/b['path'])
        if match or a['group_id']==b['group_id']:
            flags.append(dict(id=a['id'],other=b['id'],match=match))
    write(EXP/'isolation.json',dict(at=stamp(),protocol_sha256=sha(PLAN),available=len(available),
        missing=missing,flags=flags,comparisons=len(matcher.pairs),
        source_hashes={c['id']:sha(ROOT/c['path']) for c in cases},GPU_hours=0))


def panel(source,outputs,center):
    w,h = source.size
    cx,cy = center
    cw,ch = min(w,206),min(h,144)
    x,y = max(0,min(w-cw,round(w*cx-cw/2))),max(0,min(h-ch,round(h*cy-ch/2)))
    box = (x,y,x+cw,y+ch)
    out = Image.new('RGB',(1100,378),'white')
    draw = ImageDraw.Draw(out)
    for i,image in enumerate([source,*outputs]):
        ox = i*220
        tile = ImageOps.contain(image,(212,178))
        out.paste(tile,(ox+(220-tile.width)//2,24+(178-tile.height)//2))
        draw.text((ox+5,5),'Original' if not i else f'Candidate {i}',fill='black')
        draw.text((ox+5,214),'Matched native crop',fill='black')
        out.paste(image.crop(box),(ox+7,230))
    return out,box


def render(cases):
    folder = EXP/'images'
    folder.mkdir(exist_ok=True)
    rows = []
    for c in cases:
        src = ROOT/c['path']
        digest = sha(src)
        if digest!=c.get('sha256',digest):
            raise ValueError(f"Original changed: {c['id']}")
        if c['cohort']=='held':
            copy = DATA/'held-inputs'/(c['id']+src.suffix)
            copy.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,copy)
            src = copy
        rgb = load_rgb(src)
        files,edits = {},[]
        for name in RECIPES:
            result = apply(rgb,name)
            out = folder/f"{c['id']}-{name}.png"
            save(out,result)
            files[name] = dict(path=str(out.relative_to(ROOT)),sha256=sha(out))
            edits.append(Image.fromarray(result))
        comparison,box = panel(Image.fromarray(rgb),edits,c['crop'])
        p = folder/f"{c['id']}-panel.png"
        comparison.save(p)
        rows.append(dict(c,source=str(src.relative_to(ROOT)),source_sha256=digest,
            outputs=files,panel=str(p.relative_to(ROOT)),panel_sha256=sha(p),native_crop=box))
        print('Rendered',c['id'],flush=True)
    write(EXP/'replay.json',dict(at=stamp(),protocol_sha256=sha(PLAN),candidate_order=list(RECIPES),
        cases=rows,GPU_hours=0))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('render','isolate'))
    args=parser.parse_args()
    plan=read(PLAN)
    if plan['image_recipes']!={k:list(v) for k,v in RECIPES.items()}:
        raise ValueError('Recipe grid changed')
    if args.stage=='render':
        render(plan['photos'])
    else:
        isolate([c for c in plan['photos'] if c['cohort']=='held'])


if __name__=='__main__':
    main()

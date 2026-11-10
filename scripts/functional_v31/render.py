"""Render the R31 cohort using the frozen four R30 recipes."""
import argparse
from itertools import combinations
from pathlib import Path
import shutil

from PIL import Image

from functional_v1.common import ROOT,read,sha,stamp,write
from functional_v15.editor import load_rgb,save
from functional_v24.probe import SourceMatcher,protected_sources
from functional_v30.constants import RECIPES
from functional_v30.editor import apply
from functional_v30.render import panel

EXP = ROOT/'experiments/functional-v31-20261107'
DATA = ROOT/'data/training/functional-v31-20261107'
PLAN = EXP/'protocol.json'
R30 = ROOT/'experiments/functional-v30-20261107'


def isolate(cases):
    protected = protected_sources()
    for ver,name in [(24,'reviewed-candidates'),(26,'reviewed-candidates'),(27,'held-cases')]:
        for c in read(ROOT/f'data/training/functional-v{ver}-20261107/{name}.json'):
            protected[str(ROOT/c['image'])]='prior'
    for ver in (29,30):
        for c in read(ROOT/f'experiments/functional-v{ver}-20261107/replay.json')['cases']:
            protected[str(ROOT/c['source'])]='prior'
    missing={p:kind for p,kind in protected.items() if not Path(p).is_file()}
    if set(missing.values())-{'GIER_val','GIER_test'}:
        raise ValueError('Protected project photo missing')
    held=[c for c in cases if c['cohort']=='held']
    fit=[c for c in cases if c['cohort']=='fit']
    matcher,flags=SourceMatcher(),[]
    for c in held:
        other_paths={**protected,**{str(ROOT/x['path']):'new_fit' for x in fit}}
        for path,kind in other_paths.items():
            if path in missing:
                continue
            match=matcher.compare(ROOT/c['path'],path)
            if match:
                flags.append(dict(id=c['id'],other=path,kind=kind,match=match))
        print('Checked',c['id'],flush=True)
    for a,b in combinations(held,2):
        match=matcher.compare(ROOT/a['path'],ROOT/b['path'])
        if match or a['group_id']==b['group_id']:
            flags.append(dict(id=a['id'],other=b['id'],match=match))
    write(EXP/'isolation.json',dict(at=stamp(),protocol_sha256=sha(PLAN),comparisons=len(matcher.pairs),
        available=len(protected)-len(missing),missing=missing,flags=flags,
        source_hashes={c['id']:sha(ROOT/c['path']) for c in held},GPU_hours=0))


def render(cases):
    folder=EXP/'images'
    folder.mkdir(exist_ok=True)
    rows=[]
    for c in cases:
        src=ROOT/c['path']
        digest=sha(src)
        if c['cohort']=='held':
            out=DATA/'held-inputs'/(c['id']+src.suffix)
            out.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,out)
            if sha(out)!=digest:
                raise ValueError('Held original copy changed')
            src=out
        rgb=load_rgb(src)
        variants,tiles={},[]
        for name in RECIPES:
            pixels=apply(rgb,name)
            target=folder/f"{c['id']}-{name}.png"
            save(target,pixels)
            variants[name]=dict(path=str(target.relative_to(ROOT)),sha256=sha(target))
            tiles.append(Image.fromarray(pixels))
        board,box=panel(Image.fromarray(rgb),tiles,c['crop'])
        target=folder/f"{c['id']}-panel.png"
        board.save(target)
        rows.append(dict(c,source=str(src.relative_to(ROOT)),source_sha256=digest,outputs=variants,
                         panel=str(target.relative_to(ROOT)),panel_sha256=sha(target),native_crop=box))
    write(EXP/'replay.json',dict(at=stamp(),protocol_sha256=sha(PLAN),cases=rows,GPU_hours=0))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=('render','isolate'))
    args=ap.parse_args()
    cases=read(PLAN)['photos']
    (render if args.stage=='render' else isolate)(cases)


if __name__=='__main__':
    main()

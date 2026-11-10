"""Execute signed R15 comparisons on supervised and held original photographs."""
import argparse
from itertools import combinations
from pathlib import Path
import shutil

from PIL import Image

from functional_v1.common import ROOT,read,sha,stamp,write
from functional_v15.editor import load_rgb
from functional_v15.outcomes import execute
from functional_v24.probe import SourceMatcher,protected_sources
from functional_v28.boards import board
from .cache import DATA,EXP,PLAN,rows_and_plan


def isolate(photos):
    protected = protected_sources()
    for ver,name in [(24,'reviewed-candidates'),(26,'reviewed-candidates'),(27,'held-cases')]:
        for c in read(ROOT/f'data/training/functional-v{ver}-20261107/{name}.json'):
            path = ROOT/c['image']
            if sha(path)!=c['image_sha256']:
                raise ValueError(f"Prior original changed: {c['id']}")
            protected[str(path)] = 'previously_reviewed_originals'
    missing = {p:k for p,k in protected.items() if not Path(p).is_file()}
    if set(missing.values())-{'GIER_val','GIER_test'}:
        raise ValueError('A protected internal source is missing')
    match,flags = SourceMatcher(),[]
    for c in photos:
        for path,kind in protected.items():
            if path not in missing:
                evidence = match.compare(ROOT/c['path'],path)
                if evidence:
                    flags.append(dict(id=c['id'],protected=path,split=kind,evidence=evidence))
        print(f"Isolated {c['id']}",flush=True)
    for a,b in combinations(photos,2):
        evidence = match.compare(ROOT/a['path'],ROOT/b['path'])
        if evidence or a['group_id']==b['group_id']:
            flags.append(dict(id=a['id'],other_id=b['id'],evidence=evidence))
    write(EXP/'isolation.json',dict(at=stamp(),protocol_sha256=sha(PLAN),
        photos={c['id']:sha(ROOT/c['path']) for c in photos},flags=flags,
        comparisons=len(match.pairs),available=len(protected)-len(missing),missing=missing,GPU_hours=0))


def render(photos,routes):
    folder = EXP/'images'
    folder.mkdir(exist_ok=True)
    rows = []
    for c in photos:
        cid = c['id']
        path = ROOT/c['path']
        digest = sha(path)
        if digest!=c.get('sha256',digest):
            raise ValueError(f'Photo changed: {cid}')
        if c['cohort']=='held_photos':
            target = DATA/'held-inputs'/f'{cid}{path.suffix}'
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
            if sha(target)!=digest:
                raise ValueError(f'Copy changed: {cid}')
            path = target
        source = load_rgb(path)
        results,pixels = {},{}
        for name,p in routes.items():
            decision = dict(action='edit',parameters=p)
            output = execute(source,decision)['result']
            pixels[name] = output
            target = folder/f'{cid}-{name}.png'
            Image.fromarray(output).save(target)
            results[name] = dict(path=str(target.relative_to(ROOT)),sha256=sha(target),decision=decision)
        panels = {}
        for scope in ('shadow','global'):
            img,box = board(source,[pixels[scope+'_up'],pixels[scope+'_down']],c['crop'])
            target = folder/f'{cid}-{scope}-panel.png'
            img.save(target)
            panels[scope] = dict(path=str(target.relative_to(ROOT)),sha256=sha(target),crop=box)
        rows.append(dict(c,source=str(path.relative_to(ROOT)),source_sha256=digest,results=results,panels=panels))
    write(EXP/'replay.json',dict(at=stamp(),protocol_sha256=sha(PLAN),cases=rows,GPU_hours=0))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=('render','isolate'))
    args=ap.parse_args()
    _,plan=rows_and_plan()
    if args.stage=='render':
        render(plan['photos'],plan['editor_routes'])
    else:
        isolate([r for r in plan['photos'] if r['cohort']=='held_photos'])


if __name__=='__main__':
    main()

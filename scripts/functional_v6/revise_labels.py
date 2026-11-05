"""Explicit CPU editor proposals; original rejected proposals remain immutable."""
import argparse
import json
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from functional_v1.common import ROOT, read, write, sha, stamp, validate
from functional_v1.editor import load_rgb, edit, save
from functional_v1.curate import font
from .source_bank import EXP, DATA


def prepare(spec):
    out = EXP/'native-label-revision'
    assert not (out/'manifest.json').exists()
    cases = {c['id']: c for c in read(DATA/'native-training-candidates.json')}
    initial = list(map(json.loads, (EXP/'native-label-review/observations.jsonl').read_text().splitlines()))
    unresolved = {r['case_id'] for r in initial if r['selected'] is None}
    proposals = read(spec)
    assert set(proposals) == unresolved and len(initial) == 100
    rows = []
    for cid, choices in proposals.items():
        c = cases[cid]; assert sha(ROOT/c['image']) == c['input_sha256']
        rgb = load_rgb(ROOT/c['image']); candidates = {}
        assert set(choices) == {'C', 'D'}
        for code, parameters in choices.items():
            p = validate(parameters); path = out/'images'/f'{cid}-{code}.png'
            save(path, edit(rgb, p))
            candidates[code] = dict(parameters=p, image=str(path.relative_to(ROOT)), sha256=sha(path))
        rows.append(dict(case_id=cid, input_sha256=c['input_sha256'], candidates=candidates))
    sheets = []
    for offset in range(0, len(rows), 3):
        canvas = Image.new('RGB', (1800, 2220), 'white'); d = ImageDraw.Draw(canvas)
        d.text((12, 10), 'TRAINING label repair | authored proposals, not model performance', font=font(25), fill='black')
        for i, r in enumerate(rows[offset:offset+3]):
            c = cases[r['case_id']]; y = 55+i*715
            d.text((12, y), c['id'], font=font(23), fill='black')
            for j, line in enumerate(textwrap.wrap(c['instructions']['en'], 123)):
                d.text((12, y+32+j*24), line, font=font(21), fill='black')
            items = [('BEFORE', c['image'])]+[(k, v['image']) for k, v in r['candidates'].items()]
            for j, (code, path) in enumerate(items):
                x=j*600; d.text((x+12, y+90), code, font=font(23), fill='black')
                with Image.open(ROOT/path) as im:
                    fitted=ImageOps.contain(im.convert('RGB'), (575,570), Image.Resampling.LANCZOS)
                canvas.paste(fitted, (x+12+(575-fitted.width)//2,y+125+(570-fitted.height)//2))
        path=out/f'sheet-{offset//3+1:03d}.jpg';path.parent.mkdir(parents=True,exist_ok=True)
        canvas.save(path,quality=97,subsampling=0)
        sheets.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),case_ids=[r['case_id'] for r in rows[offset:offset+3]]))
    write(out/'manifest.json',dict(at=stamp(), proposals=rows,sheets=sheets,
        source_observations_sha256=sha(EXP/'native-label-review/observations.jsonl'),
        spec_sha256=sha(spec), implementation_sha256=sha(__file__),
        meaning='Root-authored parameters after rejected model proposals, executed with frozen editor. No target admission until explicit direct visual notes; this is not a blind model comparison or user approval.'))
    print('Repair proposals executed',len(rows))


def record(notes):
    out=EXP/'native-label-revision'; manifest=read(out/'manifest.json')
    assert not (out/'observations.json').exists()
    proposals={r['case_id']:r for r in manifest['proposals']}
    selected=[]
    for n in read(notes):
        assert n['case_id'] in proposals and n['language_equivalence_checked'] is True
        assert set(n['candidates'])=={'C','D'}
        for code,a in n['candidates'].items():
            assert a['completion'] in ('completed','partial','failed','uncertain')
            assert a['quality'] in ('acceptable','minor_issue','major_issue','uncertain')
            assert len(a['evidence'])>=15
            p=proposals[n['case_id']]['candidates'][code]
            assert sha(ROOT/p['image'])==p['sha256']
        code=n['selected_code'];target=None
        if code:
            a=n['candidates'][code]; assert a['completion']=='completed' and a['quality']=='acceptable'
            target=dict(proposals[n['case_id']]['candidates'][code], proposal_route='root_authored_visual_repair')
        sheet=next(s for s in manifest['sheets'] if n['case_id'] in s['case_ids']);assert sha(ROOT/sheet['path'])==sheet['sha256']
        selected.append(dict(n,selected=target,input_sha256=proposals[n['case_id']]['input_sha256'],
            sheet=sheet['path'],sheet_sha256=sheet['sha256'],at=stamp(),reviewer='Codex direct visual review, AI assisted; author-aware repair, no user approval'))
    assert len(selected)==len(proposals)==len({r['case_id'] for r in selected})
    write(out/'observations.json',selected)
    print('Repaired targets selected',sum(r['selected'] is not None for r in selected))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','record']);ap.add_argument('--spec',type=Path);ap.add_argument('--notes',type=Path)
    a=ap.parse_args()
    if a.command=='prepare':prepare(a.spec)
    else:record(a.notes)

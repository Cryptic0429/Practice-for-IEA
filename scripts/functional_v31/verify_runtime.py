"""Exercise the standalone editor on literal requests and exact source bytes."""
import numpy as np

from functional_v1.common import ROOT,read,sha,stamp,write
from functional_v15.editor import load_rgb,save
from .render import EXP
from .router import Router
from .runtime import Editor


def main():
    results=read(EXP/'result.json')
    plans={c['id']:c for c in read(EXP/'replay.json')['cases']}
    outcome={c['id']:c for c in results['cases'] if c['cohort']=='held'}
    old_route={(r['id'],r['language']):r for r in read(EXP/'route-result.json')['rows']}
    probes=[(cid,lang,plans[cid]['requests'][lang]) for cid in ('p234','p277','p475','p196') for lang in ('zh','en')]
    probes.append(('p234','control','只压暗最亮的高光，脸部保持原样。'))
    routes=Router().classify([p[2] for p in probes])
    editor=Editor()
    folder=EXP/'runtime-outputs'
    folder.mkdir(exist_ok=True)
    rows=[]
    for (cid,lang,instruction),route in zip(probes,routes):
        raw=ROOT/plans[cid]['source']
        if sha(raw)!=plans[cid]['source_sha256']:
            raise ValueError('Source changed')
        state,pixels=editor.edit(raw,instruction,route)
        target=folder/f'{cid}-{lang}.png'
        if target.exists():
            raise FileExistsError('Keep first runtime execution')
        row=dict(id=cid,language=lang,instruction=instruction,route=route,state=state,output=None)
        if lang=='control':
            if pixels is not None or state['reason']!='route_not_confirmed':
                raise ValueError('Highlight-only request entered the darkening branch')
        else:
            previous=old_route[cid,lang]
            if route['guarded']!=previous['guarded'] or max(abs(v-previous['logits'][k]) for k,v in route['logits'].items())>1e-3:
                raise ValueError('Runtime route did not reproduce scores')
            expected=outcome[cid]['chosen']['guarded_region']
            if expected['recipe']!=state.get('recipe'):
                raise ValueError('Runtime and evaluated delivery differ')
            if pixels is not None and not np.array_equal(pixels,load_rgb(ROOT/expected['output'])):
                raise ValueError('Re-executed pixels differ')
        if pixels is not None:
            save(target,pixels)
            row.update(output=str(target.relative_to(ROOT)),sha256=sha(target))
        elif target.exists():
            raise ValueError('Decline wrote an output')
        rows.append(row)
    write(EXP/'runtime-verification.json',dict(at=stamp(),result_sha256=sha(EXP/'result.json'),
        fit_sha256=sha(EXP/'fit.json'),probes=len(rows),rows=rows,GPU_hours=0,deployed=False))


if __name__=='__main__':
    main()

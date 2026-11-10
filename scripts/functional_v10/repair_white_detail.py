"""Try a shadow-weighted lift after direct white-detail concerns; no acceptance."""
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb,save
from .prepare_counterfactuals import EXP,render,INSTRUCTIONS


def main():
    out=EXP/'repairs/portrait-white-detail-07';assert not (out/'manifest.json').exists()
    c=next(c for c in read(EXP/'counterfactual-candidates/manifest.json')['cases'] if c['id']=='visual-risk-07')
    source=c['images']['canonical'];assert sha(ROOT/source['path'])==source['sha256']
    rgb=load_rgb(ROOT/source['path']);rows=[]
    for version,parameters in [('v3',dict(exposure=30,contrast=5,shadows=20,highlights=-8)),
                               ('v4',dict(exposure=25,contrast=8,shadows=35,highlights=-10))]:
        target=out/f'{version}.png';save(target,render(rgb,parameters))
        rows.append(dict(id='visual-risk-07-normal-light-'+version,parameters=parameters,
            output=dict(path=str(target.relative_to(ROOT)),sha256=sha(target)),status='executed_unreviewed_not_admitted'))
    write(out/'manifest.json',dict(at=stamp(),source=source,instructions=INSTRUCTIONS,candidates=rows,
        reason='Original and first highlight-reduced lift flattened white baby-chest detail; try more shadow-weighted lift while checking that substantial brightening still visibly occurs.',automatic_selection=False))


if __name__=='__main__':main()

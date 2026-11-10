"""Executed highlight/shadow-preserving replacements for reviewed concerns only."""
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb,save
from .prepare_counterfactuals import EXP,render
from .extend_candidates import COMPOUND


def main():
    out=EXP/'repairs/reviewed-landscape-batch-12-15-17-18-20'
    assert not (out/'manifest.json').exists()
    initial={c['id']:c for c in read(EXP/'counterfactual-candidates/manifest.json')['cases']}
    ext={c['id']:c for c in read(EXP/'extended-candidates/manifest.json')['cases']}
    style={c['id']:c for c in read(EXP/'style-candidates/manifest.json')['cases']}
    planned=[(12,'normal-compound','canonical',dict(exposure=8,contrast=8,shadows=25,saturation=12,highlights=-15)),
             (15,'normal-compound','canonical',dict(exposure=10,contrast=10,shadows=25,saturation=8,highlights=-10))]
    for index in (17,18,20):
        normal=dict(exposure=8,contrast=10,shadows=35,saturation=10)
        deep=dict(exposure=25,contrast=5,shadows=60,saturation=10)
        if index==17:normal['highlights']=-15;deep['highlights']=-10
        if index==18:
            normal=dict(exposure=8,contrast=8,shadows=20,saturation=10,highlights=-12)
            deep=dict(exposure=20,contrast=5,shadows=35,saturation=10,highlights=-12)
        planned.extend([(index,'normal-compound','canonical',normal),
                        (index,'deep-compound','deep-clean-input',deep),
                        (index,'mild-noise-compound','mild-noise-input',deep)])
    rows=[]
    for index,role,input_role,p in planned:
        cid=f'visual-risk-{index:02d}';original=style[cid] if role.startswith('normal') else ext[cid]
        source=ext[cid]['images'][input_role] if input_role=='mild-noise-input' else initial[cid]['images'][input_role]
        assert sha(ROOT/source['path'])==source['sha256']
        target=out/f'{cid}-{role}-v2.png';save(target,render(load_rgb(ROOT/source['path']),p))
        rows.append(dict(id=f'{cid}-{role}-v2',group=cid,role=role,input=source,parameters=p,
            output=dict(path=str(target.relative_to(ROOT)),sha256=sha(target)),
            original=original['images'][role],instructions=COMPOUND,status='executed_unreviewed_not_admitted'))
    write(out/'manifest.json',dict(at=stamp(),candidates=rows,
        reasons={'12':'Further flattened clouds and harsh hedge shade in normal compound.',
                 '15':'Over-intense golden light and harsh under-bridge/tree shade in normal compound.',
                 '17':'Compound made trunk and foreground shadow detail harsh/dark.',
                 '18':'Compound further flattened faint sky/cloud relief.',
                 '20':'Compound made shaded hull and wave troughs harsh/nearly black.'},
        originals_and_negative_observations_preserved=True,automatic_selection=False,L1_selection=False))
    print(f'Executed {len(rows)} landscape replacements; none admitted.')


if __name__=='__main__':main()

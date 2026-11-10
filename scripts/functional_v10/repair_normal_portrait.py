"""Executed alternatives after an original-size skin-colour concern; not automatic labels."""
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb,save
from .prepare_counterfactuals import EXP,render
from .extend_candidates import panel,COMPOUND


def main():
    out=EXP/'repairs/portrait-normal-compound-02';assert not (out/'manifest.json').exists()
    initial=read(EXP/'counterfactual-candidates/manifest.json')
    c=next(c for c in initial['cases'] if c['id']=='visual-risk-02')
    source=ROOT/c['images']['canonical']['path'];assert sha(source)==c['images']['canonical']['sha256']
    rgb=load_rgb(source);candidates=[];paths={'INPUT':source}
    for version,parameters in [('v2',dict(exposure=10,contrast=12,shadows=20,saturation=8)),
                               ('v3',dict(exposure=10,contrast=10,shadows=15,saturation=5))]:
        target=out/f'{version}.png';save(target,render(rgb,parameters));paths[version]=target
        candidates.append(dict(id='visual-risk-02-normal-compound-'+version,image=str(target.relative_to(ROOT)),sha256=sha(target),parameters=parameters,status='executed_unreviewed_not_admitted'))
    sheet=panel('visual-risk-02','normal-compound-repair',paths)
    write(out/'manifest.json',dict(at=stamp(),source=str(source.relative_to(ROOT)),input_sha256=sha(source),instructions=COMPOUND,
        reason='Contact panel judged original candidate acceptable; original-size viewing found conspicuous orange skin and strong yellow-green foliage. Keep the disagreement uncertain, exclude the old pixels from final labels and directly review new alternatives.',
        original_candidate='experiments/functional-v10-20261106/style-candidates/images/visual-risk-02-normal-compound.png',
        original_candidate_sha256=sha(EXP/'style-candidates/images/visual-risk-02-normal-compound.png'),
        panel=sheet,candidates=candidates,automatic_selection=False,L1_selection=False))
    print('Executed two new portrait alternatives; original disputed target preserved, none admitted.')

if __name__=='__main__':main()

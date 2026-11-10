"""Frozen-logit diagnosis only; no threshold search, new labels or training."""
import torch
from .common import ROOT,EXP,protocol,read,write,sha,stamp
from .head import load_head,normalize


def main():
    p=protocol();torch.set_num_threads(2);dest=EXP/'risk-repeatability-diagnostic.json';assert not dest.exists()
    source=p['hidden_manifests']['native_development'];assert sha(ROOT/source['path'])==source['sha256']
    features=read(ROOT/source['path'])['rows'];textures=read(EXP/'texture-features.json')
    cases={c['id']:c for c in read(ROOT/p['sources']['native_development']['path'])}
    parents={(r['case_id'],r['language']):r['parent_decision'] for r in read(ROOT/p['parent_semantics']['path'])['records']['native_development']}
    records=[]
    for seed,epoch in [(20261205,20),(20261205,80),(20261206,20)]:
        path=EXP/'runs'/f'seed-{seed}'/f'checkpoint-{epoch}.pt';net,item=load_head(path)
        for f in features:
            assert sha(ROOT/f['path'])==f['sha256'];cached=torch.load(ROOT/f['path'],map_location='cpu',weights_only=True)
            c=cases[f['case_id']];texture=torch.tensor(textures['images'][c['image']]['values'],dtype=torch.float32)
            x,_,_=normalize(cached['last'][None],cached['stats'][None],texture[None],item['mean'],item['std'])
            with torch.inference_mode():_,q=net(x)
            risk=float(q.sigmoid()[0]);parent=parents[c['id'],f['language']]
            records.append(dict(seed=seed,epoch=epoch,case_id=c['id'],language=f['language'],head_sha256=sha(path),
                learned_risk_sigmoid=risk,fixed_threshold=.5,parent_action=parent['action'],
                veto_applied=parent['action']=='edit' and risk>=.5,
                diagnostic_only_not_calibrated_probability=True))
    assert not torch.cuda.is_initialized()
    write(dest,dict(at=stamp(),code_sha256=sha(ROOT/'scripts/functional_v21/diagnose_risk_repeatability.py'),records=records,
        purpose='Describe why the fixed learned veto did not repeat across seeds. No threshold/label/weight changes; values are not calibrated image-quality probabilities.',
        no_new_quality_labels=True,acceptance_opened=False,GPU_calls=0))
    print([r for r in records if r['case_id']=='native-dev-070' and r['language']=='zh'])


if __name__=='__main__':main()

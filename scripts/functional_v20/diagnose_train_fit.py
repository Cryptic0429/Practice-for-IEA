"""Fixed post-fit TRAIN diagnostics only; never select or grade from these numbers."""
import torch
from .common import ROOT,EXP,ROUTES,protocol,read,write,sha,stamp
from .controller import load_controller,normalize,labels

def main():
    p=protocol();torch.set_num_threads(4)
    cases=read(ROOT/p['feature_sources']['train']['path']);a,d,m=labels([c for c in cases for _ in range(2)])
    manifest=read(EXP/'features/train-manifest.json')
    features=[torch.load(ROOT/r['path'],map_location='cpu',weights_only=True) for r in manifest['rows']]
    stats=torch.stack([r['stats'] for r in features]);results=[]
    for route in ROUTES:
        for epoch in (20,80):
            path=EXP/'runs'/route/f'checkpoint-{epoch}.pt';net,item=load_controller(path)
            x,_,_=normalize(torch.stack([r[route] for r in features]),stats,item['mean'],item['std'])
            with torch.inference_mode():aa,dd,mm=net(x)
            action=aa.argmax(-1);direction=dd.argmax(-1)
            edit=a.eq(0);active=d.ne(0)&edit[:,None]
            results.append(dict(route=route,epoch=epoch,checkpoint_sha256=sha(path),TRAIN_language_inputs=len(a),
                action_exact=int(action.eq(a).sum()),tool_direction_exact_edit_records=int(direction.eq(d).all(-1)[edit].sum()),
                action_ZH_EN_disagreement=int(action.reshape(-1,2).diff(dim=1).ne(0).sum()),
                tool_ZH_EN_disagreement_semantics=int(direction.reshape(-1,2,9).diff(dim=1).ne(0).any(-1).sum()),
                active_abs_parameter_error_diagnostic=float((mm[active]-m[active]).abs().mean()*100)))
    assert not torch.cuda.is_initialized()
    write(EXP/'TRAIN-fit-diagnostic.json',dict(at=stamp(),status='CPU_already_seen_TRAIN_only',models=results,GPU_calls=0,
        used_for_selection=False,visual_images_reviewed=0,practical_usefulness_inferred=False))
    print('TRAIN-only fit diagnostics preserved, no quality or selection inference')

if __name__=='__main__':main()

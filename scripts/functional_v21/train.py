"""One CPU head; two decoding policies share its exact weights."""
import argparse,time
import torch
from .common import ROOT,EXP,DATA,PARENT_SHA,TOOLS,protocol,read,write,sha,append,stamp
from .head import Head,labels,normalize,loss_function
from .texture import NAMES

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=[20261205,20261206],default=20261205)
    args=ap.parse_args();p=protocol();torch.set_num_threads(4)
    assert read(EXP/'cpu-preflight.json')['status']=='passed'
    assert not torch.cuda.is_initialized()
    cases=read(DATA/'cases.json');assert len(cases)==925
    source=p['hidden_manifests']['train'];assert sha(ROOT/source['path'])==source['sha256']
    manifest=read(ROOT/source['path']);rows=manifest['rows']
    assert [(r['case_id'],r['language']) for r in rows]==[(c['id'],l) for c in cases for l in ('zh','en')]
    cached=[]
    for row in rows:
        assert sha(ROOT/row['path'])==row['sha256']
        item=torch.load(ROOT/row['path'],map_location='cpu',weights_only=True)
        assert item['protocol_sha256']==p['hidden_protocol_sha256'];cached.append(item)
    textures=read(EXP/'texture-features.json')
    assert textures['implementation_sha256']==sha(ROOT/'scripts/functional_v21/texture.py')
    texture=torch.tensor([textures['images'][c['image']]['values'] for c in cases for _ in range(2)],dtype=torch.float32)
    x,mean,std=normalize(torch.stack([r['last'] for r in cached]),torch.stack([r['stats'] for r in cached]),texture)
    targets=labels([c for c in cases for _ in range(2)])
    magnitude,active,quality,supported=targets
    pos_weight=((supported.sum()-quality.sum())/quality.sum()).sqrt().clamp(1,5)
    torch.manual_seed(args.seed);net=Head();initial={k:v.clone() for k,v in net.state_dict().items()}
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    generator=torch.Generator().manual_seed(args.seed)
    out=EXP/'runs'/f'seed-{args.seed}';out.mkdir(parents=True,exist_ok=False)
    write(out/'settings.json',dict(seed=args.seed,epochs=80,steps_per_epoch=58,policies_share_weights=True,
        parameters=sum(v.numel() for v in net.parameters()),semantic_cases=925,language_inputs=1850,
        quality_language_positive=int(quality.sum()),quality_language_supervised=int(supported.sum()),
        quality_pos_weight=float(pos_weight),normalization_TRAIN_only=True,hidden_manifest_sha256=source['sha256'],texture_sha256=sha(EXP/'texture-features.json')))
    start=time.monotonic();steps=0
    for epoch in range(1,81):
        net.train();order=torch.randperm(len(cases),generator=generator);losses=[]
        for offset in range(0,len(cases),16):
            ids=order[offset:offset+16];indices=torch.stack([2*ids,2*ids+1],-1).flatten()
            loss=loss_function(net(x[indices]),*(t[indices] for t in targets),pos_weight)
            assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in net.parameters())
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step();steps+=1;losses.append(float(loss.detach()))
        append(out/'metrics.jsonl',dict(epoch=epoch,step=steps,loss=sum(losses)/len(losses),diagnostic_only=True))
        if epoch in (20,80):
            torch.save(dict(state_dict={k:v.detach().clone() for k,v in net.state_dict().items()},mean=mean,std=std,
                tools=list(TOOLS),texture_names=list(NAMES),epoch=epoch,step=steps,parent_sha256=PARENT_SHA,
                protocol_sha256=sha(EXP/'training-protocol.json'),cases_sha256=p['sources']['train']['sha256']),out/f'checkpoint-{epoch}.pt')
            torch.save(dict(optimizer=optimizer.state_dict(),generator=generator.get_state(),torch_rng=torch.get_rng_state()),out/f'training-state-{epoch}.pt')
        write(out/'status.json',dict(status='running',epoch=epoch,step=steps,cpu_only=True,at=stamp()))
    assert all(not torch.equal(initial[k],v) for k,v in net.state_dict().items())
    assert not torch.cuda.is_initialized()
    write(out/'status.json',dict(status='completed',epoch=80,step=steps,cpu_only=True,all_weights_changed=True,
        wall_seconds=time.monotonic()-start,checkpoint_sha256={str(e):sha(out/f'checkpoint-{e}.pt') for e in (20,80)},at=stamp()))
    print(read(out/'status.json'))

if __name__=='__main__':main()

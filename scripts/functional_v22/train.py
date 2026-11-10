"""Same R21 head/loss/optimizer, paired seeds and exactly1160 CPU updates."""
import argparse,time
import torch
from .common import ROOT,EXP,DATA,PARENT_SHA,TOOLS,protocol,read,write,sha,append,stamp
from functional_v21.head import Head,labels,normalize,loss_function
from functional_v21.texture import NAMES

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=[20261205,20261206],required=True);args=ap.parse_args()
    p=protocol();torch.set_num_threads(4);assert not torch.cuda.is_initialized()
    cases=read(DATA/'cases.json');manifest=read(EXP/'features/train-manifest.json');assert manifest['source_sha256']==sha(DATA/'cases.json')
    rows=manifest['rows'];assert [(r['case_id'],r['language']) for r in rows]==[(c['id'],l) for c in cases for l in ('zh','en')]
    cached=[]
    for r in rows:
        c=cases[len(cached)//2];assert sha(ROOT/r['path'])==r['sha256'];v=torch.load(ROOT/r['path'],map_location='cpu',weights_only=True)
        assert v['protocol_sha256']==r['feature_protocol_sha256'] and v['input_sha256']==c['input_sha256'] and v['request']==c['instructions'][r['language']]
        cached.append(v)
    texture=read(EXP/'texture-features.json');assert texture['implementation_sha256']==sha(ROOT/'scripts/functional_v21/texture.py')
    tex=torch.tensor([texture['images'][c['image']]['values'] for c in cases for _ in range(2)],dtype=torch.float32)
    x,mean,std=normalize(torch.stack([r['last'] for r in cached]),torch.stack([r['stats'] for r in cached]),tex)
    targets=labels([c for c in cases for _ in range(2)]);magnitude,active,quality,supported=targets
    pos_weight=((supported.sum()-quality.sum())/quality.sum()).sqrt().clamp(1,5)
    torch.manual_seed(args.seed);net=Head();initial={k:v.clone() for k,v in net.state_dict().items()}
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01);generator=torch.Generator().manual_seed(args.seed)
    out=EXP/'runs'/f'seed-{args.seed}';out.mkdir(parents=True,exist_ok=False)
    write(out/'settings.json',dict(seed=args.seed,updates=1160,parameters=sum(v.numel() for v in net.parameters()),semantic_condition_cases=len(cases),language_records=len(rows),
        quality_language_positive=int(quality.sum()),quality_language_supervised=int(supported.sum()),quality_pos_weight=float(pos_weight),
        normalization_TRAIN_only=True,initial_state_sha256s={k:__import__('hashlib').sha256(v.numpy().tobytes()).hexdigest() for k,v in initial.items()},
        train_manifest_sha256=sha(EXP/'features/train-manifest.json'),texture_sha256=sha(EXP/'texture-features.json')))
    start=time.monotonic();step=presentations=0;epoch=0
    while step<1160:
        epoch+=1;net.train();order=torch.randperm(len(cases),generator=generator)
        for offset in range(0,len(cases),16):
            if step==1160:break
            ids=order[offset:offset+16];indices=torch.stack([2*ids,2*ids+1],-1).flatten()
            loss=loss_function(net(x[indices]),*(t[indices] for t in targets),pos_weight)
            assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in net.parameters())
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step();step+=1;presentations+=len(indices)
            if step%20==0:append(out/'metrics.jsonl',dict(step=step,pass_number=epoch,language_presentations=presentations,loss=float(loss.detach()),diagnostic_only=True))
        write(out/'status.json',dict(status='running',step=step,pass_number=epoch,language_presentations=presentations,cpu_only=True,at=stamp()))
    assert step==1160 and all(not torch.equal(initial[k],v) for k,v in net.state_dict().items())
    assert all(torch.isfinite(v).all() for v in net.state_dict().values()) and not torch.cuda.is_initialized()
    net.eval();torch.save(dict(state_dict={k:v.detach().clone() for k,v in net.state_dict().items()},mean=mean,std=std,tools=list(TOOLS),texture_names=list(NAMES),
        step=step,completed_full_passes=epoch-1,partial_pass=epoch,language_presentations=presentations,parent_sha256=PARENT_SHA,
        protocol_sha256=sha(EXP/'training-protocol.json'),cases_sha256=sha(DATA/'cases.json')),out/'checkpoint-step-1160.pt')
    torch.save(dict(optimizer=optimizer.state_dict(),generator=generator.get_state(),torch_rng=torch.get_rng_state(),step=step,order=order,offset=offset),out/'training-state-step-1160.pt')
    write(out/'status.json',dict(status='completed',step=step,completed_full_passes=epoch-1,partial_pass=epoch,language_presentations=presentations,cpu_only=True,
        all4tensors_changed_finite=True,wall_seconds=time.monotonic()-start,checkpoint_sha256=sha(out/'checkpoint-step-1160.pt'),
        optimizer_RNG_sha256=sha(out/'training-state-step-1160.pt'),at=stamp()))
    print(read(out/'status.json'))

if __name__=='__main__':main()

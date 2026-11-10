"""Inspect real checkpoints/optimizer/RNG and recompute TRAIN-only normalization."""
import argparse
import torch
from .common import ROOT,EXP,DATA,protocol,read,write,sha,stamp
from .head import Head,load_head,normalize

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=[20261205,20261206],default=20261205);args=ap.parse_args()
    p=protocol();torch.set_num_threads(4);cases=read(DATA/'cases.json')
    source=p['hidden_manifests']['train'];assert sha(ROOT/source['path'])==source['sha256']
    cached=[]
    for row in read(ROOT/source['path'])['rows']:
        assert sha(ROOT/row['path'])==row['sha256']
        cached.append(torch.load(ROOT/row['path'],map_location='cpu',weights_only=True))
    textures=read(EXP/'texture-features.json')
    _,mean,std=normalize(torch.stack([r['last'] for r in cached]),torch.stack([r['stats'] for r in cached]),
        torch.tensor([textures['images'][c['image']]['values'] for c in cases for _ in range(2)],dtype=torch.float32))
    checks=[]
    for epoch in (20,80):
        folder=EXP/'runs'/f'seed-{args.seed}';path=folder/f'checkpoint-{epoch}.pt';model,item=load_head(path)
        assert item['protocol_sha256']==sha(EXP/'training-protocol.json')
        assert torch.equal(mean,item['mean']) and torch.equal(std,item['std'])
        assert item['epoch']==epoch and item['step']==58*epoch
        torch.manual_seed(args.seed);initial=Head()
        assert all(t.dtype==torch.float32 and torch.isfinite(t).all() and not torch.equal(t,initial.state_dict()[name]) for name,t in item['state_dict'].items())
        state=torch.load(folder/f'training-state-{epoch}.pt',map_location='cpu',weights_only=True)
        assert len(state['optimizer']['state'])==4 and set(state)=={'optimizer','generator','torch_rng'}
        assert all(torch.isfinite(t).all() for value in state['optimizer']['state'].values() for t in value.values() if torch.is_tensor(t))
        checks.append(dict(epoch=epoch,step=item['step'],trainable_parameters=sum(v.numel() for v in model.parameters()),
            head_sha256=sha(path),optimizer_RNG_sha256=sha(folder/f'training-state-{epoch}.pt'),
            finite_and_changed_4tensors=True,exact_TRAIN_only_normalization=True))
    assert not torch.cuda.is_initialized()
    destination=EXP/('training-verification.json' if args.seed==20261205 else 'replication-training-verification.json')
    assert not destination.exists(),'Preserve existing verification receipts'
    write(destination,dict(at=stamp(),seed=args.seed,status='actual_CPU_training_verified',checkpoints=checks,
        independently_trained_heads=1,decoding_policies=2,policies_share_weights=True,language_presentations=148000,
        GPU_calls=0,quality_or_practical_usefulness_inferred=False))
    print('Verified2 actual checkpoints ofone shared head, optimizer/RNG, TRAIN normalizer; noGPU orquality inference.')

if __name__=='__main__':main()

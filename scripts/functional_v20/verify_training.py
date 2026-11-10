"""Inspect actual CPU head checkpoints/state and train-fitted normalizers."""
import torch
from .common import ROOT,EXP,ROUTES,protocol,read,write,sha,stamp
from .controller import Controller,load_controller,normalize

def main():
    p=protocol();torch.set_num_threads(4)
    manifest=read(EXP/'features/train-manifest.json');cached=[]
    for row in manifest['rows']:
        assert sha(ROOT/row['path'])==row['sha256']
        cached.append(torch.load(ROOT/row['path'],map_location='cpu',weights_only=True))
    stats=torch.stack([r['stats'] for r in cached]);checks=[]
    for route in ROUTES:
        _,mean,std=normalize(torch.stack([r[route] for r in cached]),stats)
        for epoch in (20,80):
            folder=EXP/'runs'/route;path=folder/f'checkpoint-{epoch}.pt'
            model,item=load_controller(path)
            assert item['protocol_sha256']==sha(EXP/'training-protocol.json')
            assert torch.equal(mean,item['mean']) and torch.equal(std,item['std'])
            assert item['epoch']==epoch and item['step']==58*epoch
            assert all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in item['state_dict'].values())
            torch.manual_seed(p['training']['seed']);initial=Controller()
            assert all(not torch.equal(t,initial.state_dict()[name]) for name,t in item['state_dict'].items())
            state=torch.load(folder/f'training-state-{epoch}.pt',map_location='cpu',weights_only=True)
            assert len(state['optimizer']['state'])==4
            assert all(torch.isfinite(v).all() for group in state['optimizer']['state'].values() for v in group.values() if torch.is_tensor(v))
            checks.append(dict(route=route,epoch=epoch,step=item['step'],parameters=sum(v.numel() for v in model.parameters()),
                checkpoint_sha256=sha(path),optimizer_state_sha256=sha(folder/f'training-state-{epoch}.pt'),
                finite_tensors=True,all4_weight_tensors_changed=True,normalizer_exact_TRAIN_only=True))
    assert not torch.cuda.is_initialized()
    write(EXP/'training-verification.json',dict(at=stamp(),status='actual_CPU_training_verified',checkpoints=checks,
        language_presentations_per_route=1850*80,GPU_calls=0,quality_inferred_from_fit=False))
    print('Verified4 actual heads, optimizer/RNG and exact TRAIN-only normalization; no functional claim')

if __name__=='__main__':main()

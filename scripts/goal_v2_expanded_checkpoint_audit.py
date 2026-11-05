"""Audit own generated checkpoint state before any explicit continuation."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import load_file

ROOT=Path('/mnt/sdc/zhc/IEA')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint');parser.add_argument('--step',type=int,required=True)
    args=parser.parse_args();cp=Path(args.checkpoint).resolve()
    assert cp.is_relative_to(ROOT/'outputs/goal-v2-20261103')
    state=json.loads((cp/'trainer_state.json').read_text());assert state['global_step']==args.step
    weights=load_file(str(cp/'adapter_model.safetensors'))
    assert len(weights)==336 and all('lora_' in k and torch.isfinite(v).all() for k,v in weights.items())
    optimizer=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=True)
    expected={p for group in optimizer['param_groups'] for p in group['params']}
    assert len(expected)==336 and set(optimizer['state'])==expected
    for state in optimizer['state'].values():
        assert int(state['step'])==args.step
        assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
    scheduler=torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=True)
    assert scheduler['last_epoch']==args.step
    # This pickle is restricted to checkpoints created by our own trainer under this experiment.
    rng=torch.load(cp/'rng_state.pth',map_location='cpu',weights_only=False)
    assert all(k in rng for k in ['python','numpy','cpu','cuda'])
    files=['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json']
    report=dict(passed=True,step=args.step,adapter_tensors=len(weights),optimizer_parameter_states=len(expected),
                scheduler_last_epoch=scheduler['last_epoch'],rng_components=list(rng),
                files={name:hashlib.sha256((cp/name).read_bytes()).hexdigest() for name in files},
                scope='State existence,identity,step alignment and finiteness; not an executed resume equivalence test.')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()

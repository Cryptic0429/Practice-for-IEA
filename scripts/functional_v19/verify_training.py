"""Inspect actual completed checkpoint weights, optimizer, schedule and RNG on CPU."""
import hashlib
import torch
from safetensors.torch import load_file
from functional_v1.common import ROOT,read,write,sha,stamp
EXP=ROOT/'experiments/functional-v19-20261107'

def main():
    target=EXP/'training-verification.json';assert not target.exists()
    protocol=read(EXP/'training-protocol.json');run=EXP/'runs/photometric_bridge-seed-20261129';state=read(run/'status.json')
    assert state['status']=='completed' and state['step']==1002
    assert all(sha(ROOT/name)==digest for name,digest in protocol['code_hashes'].items())
    initial=read(run/'initial-adapter-fingerprints.json')
    normalized={key.replace('.modules_to_save.default','').replace('.default',''):digest for key,digest in initial.items()}
    assert len(normalized)==120
    checks=[]
    for step in (501,1002):
        folder=run/f'checkpoint-{step}';assert read(folder/'trainer_state.json')['global_step']==step
        weights=load_file(str(folder/'adapter_model.safetensors'),device='cpu')
        assert len(weights)==120 and set(weights)==set(normalized)
        assert all(value.dtype==torch.float32 and torch.isfinite(value).all() for value in weights.values())
        assert sum(value.numel() for value in weights.values())==3139200
        optimizer=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=True)
        assert len(optimizer['state'])==120
        groups=optimizer['param_groups'];assert len(groups)==2 and [len(g['params']) for g in groups]==[116,4]
        assert [g['initial_lr'] for g in groups]==[1e-5,1e-3]
        assert abs(groups[1]['lr']-100*groups[0]['lr'])<1e-12
        assert all(torch.isfinite(value).all() for item in optimizer['state'].values() for value in item.values() if torch.is_tensor(value))
        scheduler=torch.load(folder/'scheduler.pt',map_location='cpu',weights_only=True)
        assert scheduler['last_epoch']==step and scheduler['base_lrs']==[1e-5,1e-3]
        # This RNG file was produced by our own just-completed Trainer, not an
        # external checkpoint. NumPy RNG tuples require ordinary deserialization.
        rng=torch.load(folder/'rng_state.pth',map_location='cpu',weights_only=False)
        assert {'python','numpy','cpu','cuda'}<=set(rng)
        assert torch.is_tensor(rng['cpu']) and torch.is_tensor(rng['cuda']) and rng['cpu'].numel()>0 and rng['cuda'].numel()>0
        changed=sum(hashlib.sha256(value.numpy().tobytes()).hexdigest()!=normalized[key] for key,value in weights.items())
        checks.append(dict(step=step,saved_tensors=120,parameters=3139200,changed_since_parent_initialization=changed,
            optimizer_groups=[dict(tensors=len(g['params']),initial_lr=g['initial_lr'],actual_lr=g['lr']) for g in groups],
            optimizer_tensors_finite=True,scheduler_step=step,RNG_keys=sorted(rng),
            files={name:dict(bytes=(folder/name).stat().st_size,sha256=sha(folder/name)) for name in ('adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json')}))
    assert checks[-1]['changed_since_parent_initialization']==120
    assert sha(run/'final/adapter_model.safetensors')==checks[-1]['files']['adapter_model.safetensors']['sha256']==state['adapter_sha256']
    assert state['actual_objective_calls']==dict(anchor=3700,preference=304)
    assert not torch.cuda.is_initialized()
    write(target,dict(at=stamp(),status='actual_completed_training_artifacts_verified_CPU',checkpoints=checks,
        training_protocol_sha256=sha(EXP/'training-protocol.json'),optimizer_steps=1002,positive_presentations=4004,
        rejected_forwards=0,one_adapter_including_bridge=True,GPU_calls=0,model_functional_success_inferred=False))
    print('Verified both actual120tensor checkpoints,3139200parameters, optimizer116+4 groups, schedule and Python/NumPy/CPU/CUDA RNG; no GPU or quality claim.')

if __name__=='__main__':main()

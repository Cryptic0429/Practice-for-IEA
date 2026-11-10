"""CPU-only frozen-checkpoint branch variation; never an editing-quality score."""
import argparse
import torch
from safetensors.torch import load_file
from functional_v1.common import ROOT,read,write,sha,stamp
from .bridge import StatBridge
EXP=ROOT/'experiments/functional-v19-20261107'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',type=int,choices=(1,2),required=True);args=parser.parse_args()
    protocol=read(EXP/'bridge-audit-protocol.json');training=read(EXP/'training-protocol.json')
    assert sha(ROOT/'scripts/functional_v19/audit_bridge.py')==protocol['audit_source_sha256']
    assert sha(EXP/'training-protocol.json')==protocol['training_protocol_sha256']
    assert sha(ROOT/training['input_features_path'])==training['input_features_sha256']
    checkpoint=EXP/'runs/photometric_bridge-seed-20261129'/f'checkpoint-{501*args.epoch}'
    assert read(checkpoint/'trainer_state.json')['global_step']==501*args.epoch
    out=EXP/'bridge-audit'/f'epoch{args.epoch}.json';assert not out.exists()
    weights=checkpoint/'adapter_model.safetensors';digest=sha(weights);saved=load_file(str(weights),device='cpu')
    assert len(saved)==120
    bridge_state={k.split('.stat_bridge.',1)[1]:v for k,v in saved.items() if '.stat_bridge.' in k}
    assert len(bridge_state)==4
    torch.set_num_threads(2);model=StatBridge(3584).float().eval();model.load_state_dict(bridge_state,strict=True)
    cases=read(ROOT/training['input_features_path'])['cases'];distinct={}
    for case in cases.values():
        if case['input_sha256'] in distinct:assert distinct[case['input_sha256']]==case['features']
        distinct[case['input_sha256']]=case['features']
    features=torch.tensor(list(distinct.values()),dtype=torch.float32)
    with torch.no_grad():deltas=torch.cat([model(row[None]) for row in features])
    mean=deltas.mean(0);centered=deltas-mean
    total_square=float(deltas.square().mean());mean_square=float(mean.square().mean());variable_square=float(centered.square().mean())
    assert abs(total_square-mean_square-variable_square)<1e-5*max(1,total_square)
    assert not torch.cuda.is_initialized()
    write(out,dict(at=stamp(),scope='CPU FP32 reconstruction of the small branch on469 unique TRAIN input-byte identities; not whole-model inference or editing quality.',
        checkpoint=str(checkpoint.relative_to(ROOT)),adapter_sha256=digest,training_protocol_sha256=sha(EXP/'training-protocol.json'),
        input_features_sha256=training['input_features_sha256'],semantic_cases=len(cases),unique_inputs=len(distinct),
        branch_output_RMS=total_square**.5,common_mean_component_RMS=mean_square**.5,input_variable_component_RMS=variable_square**.5,
        fraction_of_branch_squared_norm_due_to_input_variation=variable_square/total_square if total_square else None,
        last_bias_RMS=float(model.last.bias.square().mean().sqrt()),finite=bool(torch.isfinite(deltas).all()),
        weight_norms={name:float(value.norm()) for name,value in bridge_state.items()},
        GPU_calls=0,checkpoint_selection_used=False,training_or_data_changed=False,
        limitation='Input-dependent branch activations do not prove the7B decoder uses them correctly. A common component does not prove collapse. CPU FP32 values are not a claim of bit-identical CUDA/autocast numerics. Functional visual gates remain authoritative.'))
    assert sha(weights)==digest
    print('CPU branch audit saved',out.relative_to(ROOT),'; no GPU or functional score.')

if __name__=='__main__':main()

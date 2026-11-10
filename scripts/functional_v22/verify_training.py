"""Verify both actual runs, TRAIN-only normalization and target-free row policy."""
import copy,hashlib
import torch
from .common import ROOT,EXP,DATA,PARENT_SHA,protocol,read,write,sha,stamp
from functional_v21.head import Head,normalize,load_head
from functional_v15.outcomes import input_row

def main():
    assert not (EXP/'training-verification.json').exists();p=protocol();torch.set_num_threads(4)
    cases=read(DATA/'cases.json');rows=read(EXP/'features/train-manifest.json')['rows'];texture=read(EXP/'texture-features.json')
    saved=[torch.load(ROOT/r['path'],map_location='cpu',weights_only=True) for r in rows]
    x,mean,std=normalize(torch.stack([v['last'] for v in saved]),torch.stack([v['stats'] for v in saved]),
        torch.tensor([texture['images'][c['image']]['values'] for c in cases for _ in range(2)],dtype=torch.float32))
    records=[]
    for seed in p['training']['seeds']:
        folder=EXP/'runs'/f'seed-{seed}';status=read(folder/'status.json');settings=read(folder/'settings.json')
        assert status['status']=='completed' and status['step']==1160 and status['language_presentations']==36652
        checkpoint=folder/'checkpoint-step-1160.pt';net,item=load_head(checkpoint)
        assert sha(checkpoint)==status['checkpoint_sha256'] and item['protocol_sha256']==sha(EXP/'training-protocol.json')
        assert item['parent_sha256']==PARENT_SHA and item['cases_sha256']==sha(DATA/'cases.json')
        assert torch.equal(item['mean'],mean) and torch.equal(item['std'],std)
        torch.manual_seed(seed);initial=Head().state_dict();current=net.state_dict()
        assert len(current)==4 and sum(v.numel() for v in current.values())==463754
        assert all(v.dtype==torch.float32 and torch.isfinite(v).all() and not torch.equal(initial[k],v) for k,v in current.items())
        assert settings['initial_state_sha256s']=={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in initial.items()}
        recovery=folder/'training-state-step-1160.pt';state=torch.load(recovery,map_location='cpu',weights_only=True)
        assert sha(recovery)==status['optimizer_RNG_sha256'] and state['step']==1160 and len(state['optimizer']['state'])==4
        for value in state['optimizer']['state'].values():
            assert int(value['step'])==1160
            assert all(torch.isfinite(t).all() for t in value.values() if torch.is_tensor(t))
        assert state['torch_rng'].numel()>0 and state['generator'].numel()>0
        assert torch.equal(state['order'].sort().values,torch.arange(len(cases)))
        assert int(state['offset'])==704
        with torch.inference_mode():a=net(x[:8]);b=net(x[:8])
        assert all(torch.equal(y,z) and torch.isfinite(y).all() for y,z in zip(a,b))
        records.append(dict(seed=seed,step=1160,language_presentations=36652,parameters=463754,finite_changed_F32_tensors=4,
            checkpoint_sha256=sha(checkpoint),optimizer_RNG_sha256=sha(recovery),TRAIN_normalization_exact=True,recovery_state_verified=True))
    case=cases[-1];baseline=input_row(case,'zh');changed=copy.deepcopy(case)
    changed.update(target='/never/read/target.png',label_evidence='untrusted metadata not part of prompt',family='poison_category',id='poison_case',
        condition_origin={'reference':'must not enter encoder'},decision={'action':'keep'})
    assert input_row(changed,'zh')==baseline
    assert len(baseline['conversations'])==2 and [v['role'] for v in baseline['conversations']]==['system','user']
    assert baseline['conversations'][1]['content']=='<image>\n'+case['instructions']['zh']
    assert not torch.cuda.is_initialized()
    write(EXP/'training-verification.json',dict(at=stamp(),status='both_actual_CPU_training_runs_verified',records=records,
        combined_language_presentations=73304,actual_GPU_training_hours=0,input_metadata_poison_test='passed_exact_input_row_unchanged',
        no_quality_improvement_inferred=True,independent_user_feedback=0))
    print('Verified two actual1160-step heads/73304 language presentations, all4 finite changed tensors each, exact TRAIN normalization, recovery and input whitelist; not a quality score.')

if __name__=='__main__':main()

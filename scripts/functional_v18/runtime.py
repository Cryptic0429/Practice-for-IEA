"""One R15 parent adapter, shared by the matched SFT/ranking routes."""
from functional_v1.common import ROOT, read, sha
from functional_v15 import runtime as previous

EXP = ROOT/'experiments/functional-v18-20261107'
DATA = ROOT/'data/training/functional-v18-20261107'
PARENT = ROOT/'models/from-functional-v11/sft-shadow-correction/seed-01'
PARENT_SHA = '97c84f1799fb319799c86515402574b3b585593763e5388eb30d0df3e0d6ea90'
ROUTES = ('counterfactual_rank', 'matched_sft')
processor = previous.processor
build_batch = previous.build_batch
predict = previous.predict


def expected_count(route):
    assert route in ROUTES
    return 116


def verify_origin():
    identity=previous.verify_origin()
    assert sha(PARENT/'adapter_model.safetensors')==PARENT_SHA
    assert read(PARENT/'adapter_config.json')['r']==8
    return identity


def load_model(route, adapter=None, train=False, seed=None):
    assert route in ROUTES
    verify_origin()
    if not train:
        return previous.load_model('language_merger',adapter=adapter or PARENT)
    assert adapter is None and seed is not None
    import torch
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    from functional_v7.runtime import canonical
    model=previous.load_model('language_merger',train=True,seed=seed)
    loader_initialization=dict(model.functional_initialization)
    saved=load_file(str(PARENT/'adapter_model.safetensors'),device='cpu')
    wanted={canonical(k):v for k,v in saved.items()}
    actual=get_peft_model_state_dict(model)
    assert len(actual)==len(saved)==len(wanted)==116
    assert {canonical(k) for k in actual}==set(wanted)
    copied={k:wanted[canonical(k)].to(v.dtype) for k,v in actual.items()}
    result=set_peft_model_state_dict(model,copied)
    assert not result.unexpected_keys
    after=get_peft_model_state_dict(model)
    assert all(torch.equal(v.detach().cpu(),copied[k].cpu()) for k,v in after.items())
    assert len(model.peft_config)==1
    assert len([p for p in model.parameters() if p.requires_grad])==116
    model.functional_initialization=dict(origin='base',parent_path=str(PARENT.relative_to(ROOT)),parent_sha256=PARENT_SHA,
        parent_tensors_copied_exactly=116,language_parent_tensors_copied_exactly=112,
        visual_parent_tensors_copied_exactly=4,adapter_count=1,stacked_adapters=False,
        loader_initialization_before_full_R15_replacement=loader_initialization,
        numerical_path='Exact R13/R15 NF4 kbit-prepared training path. All adapter tensors then replaced by R15e1. Parameter identity does not assert identical BF16/FP32 inference logits.')
    return model

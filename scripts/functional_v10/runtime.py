"""Single-parent adapter correction with optional visual merger; frozen R7 tools."""
from pathlib import Path
import torch
from functional_v1.common import ROOT, BASE, read, sha
from functional_v1.runtime import processor, encode, generate, verify_adapter
from functional_v7.runtime import build_batch, create_adapter, transfer_parent, set_trainable, predict, ROUTES
from functional_v7.outcomes import input_row, target_text, parse
from functional_v9.runtime import verify_origin as verify_original_origin

EXP=ROOT/'experiments/functional-v10-20261106'
DATA=ROOT/'data/training/functional-v10-20261106'
PARENT=ROOT/'experiments/functional-v9-20261106/runs/base-seed-20261115/checkpoint-231'
PARENT_SHA='dc92eb24484a383d71a7b8cc9cddca6a0c432e6c32804d4f2f31996376230d3e'

def verify_origin():
    identity=verify_original_origin('base')
    assert sha(PARENT/'adapter_model.safetensors')==PARENT_SHA
    assert Path(read(PARENT/'adapter_config.json')['base_model_name_or_path']).resolve()==BASE.resolve()
    return identity

def load_model(route,adapter=None,train=False,seed=None):
    from transformers import BitsAndBytesConfig,Qwen2_5_VLForConditionalGeneration,set_seed
    from peft import PeftModel,prepare_model_for_kbit_training
    assert route in ROUTES
    assert not train or (adapter is None and seed is not None)
    verify_origin()
    base,information=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':0},output_loading_info=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    assert not any(information.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),information
    if train:
        base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
        set_seed(seed)
        model=create_adapter(base,route)
        transfer=transfer_parent(model,PARENT,route)
        set_trainable(model,route)
        model.functional_initialization=dict(origin='base',parent_path=str(PARENT.relative_to(ROOT)),parent_sha256=PARENT_SHA,
            full_weight_loading_information=information,**transfer)
    else:
        assert adapter is not None,'Inference must specify its actual trained adapter'
        adapter=Path(adapter)
        assert Path(read(adapter/'adapter_config.json')['base_model_name_or_path']).resolve()==BASE.resolve()
        model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False)
        verify_adapter(model,adapter)
        for weight in model.parameters():weight.requires_grad_(False)
        model.eval()
    return model

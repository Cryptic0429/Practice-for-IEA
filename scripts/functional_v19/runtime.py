"""R15 parent LoRA plus a trainable input-RGB bridge, saved in the same PEFT adapter."""
from pathlib import Path
import torch
from functional_v1.common import ROOT,BASE,read,sha
from functional_v18 import runtime as parent_runtime
from functional_v15 import runtime as previous
from functional_v7.runtime import canonical,MERGER_PATTERN
from .bridge import install_before_peft,set_features
from .photometry import features

EXP=ROOT/'experiments/functional-v19-20261107'
DATA=ROOT/'data/training/functional-v19-20261107'
PARENT=parent_runtime.PARENT
PARENT_SHA=parent_runtime.PARENT_SHA
ROUTES=('photometric_bridge',)
processor=previous.processor
build_batch=previous.build_batch
verify_origin=parent_runtime.verify_origin

def expected_count(route):
    assert route in ROUTES
    return 120

def load_model(route,adapter=None,train=False,seed=None):
    assert route in ROUTES and (not train or (adapter is None and seed is not None))
    verify_origin()
    from transformers import BitsAndBytesConfig,Qwen2_5_VLForConditionalGeneration,set_seed
    from peft import LoraConfig,get_peft_model,PeftModel,prepare_model_for_kbit_training,get_peft_model_state_dict,set_peft_model_state_dict
    from safetensors.torch import load_file
    base,information=Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',device_map={'':0},output_loading_info=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    assert not any(information.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),information
    if train:
        base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
        set_seed(seed)
    install_before_peft(base,base.config.text_config.hidden_size)
    if train:
        model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,bias='none',task_type='CAUSAL_LM',
            target_modules=MERGER_PATTERN,modules_to_save=['stat_bridge']))
        saved=load_file(str(PARENT/'adapter_model.safetensors'),device='cpu');wanted={canonical(k):v for k,v in saved.items()}
        actual=get_peft_model_state_dict(model);assert len(saved)==116 and len(actual)==120
        copied={k:wanted[canonical(k)].to(v.dtype) if canonical(k) in wanted else v for k,v in actual.items()}
        result=set_peft_model_state_dict(model,copied);assert not result.unexpected_keys
        after=get_peft_model_state_dict(model)
        assert all(torch.equal(after[k].detach().cpu(),wanted[canonical(k)].to(after[k].dtype)) for k in after if canonical(k) in wanted)
        assert sum(canonical(k) in wanted for k in after)==116
        active=[]
        for name,value in model.named_parameters():
            enabled='lora_' in name or '.stat_bridge.modules_to_save.default.' in name
            value.requires_grad_(enabled)
            if enabled:value.data=value.data.float();active.append(name)
        assert len(active)==120 and sum('.stat_bridge.' in n for n in active)==4
        model.config.use_cache=False;model.train()
        model.functional_initialization=dict(origin='base',parent_path=str(PARENT.relative_to(ROOT)),parent_sha256=PARENT_SHA,
            parent_tensors_copied_exactly=116,language_parent_tensors_copied_exactly=112,visual_parent_tensors_copied_exactly=4,
            added_bridge_tensors=4,bridge_output_layer_zero=True,bridge_input_dimension=16,bridge_hidden_dimension=128,
            bridge_output_dimension=base.config.text_config.hidden_size,adapter_count=1,stacked_adapters=False,
            full_weight_loading_information=information,numerical_path='Same NF4/kbit-prepared path as R18; one zero-output input-RGB bridge. Greedy parity verified separately before updates.')
    else:
        assert adapter is not None
        config=read(Path(adapter)/'adapter_config.json')
        assert Path(config['base_model_name_or_path']).resolve()==BASE.resolve()
        assert config['target_modules']==MERGER_PATTERN and config['modules_to_save']==['stat_bridge']
        model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False)
        state=get_peft_model_state_dict(model);saved=load_file(str(Path(adapter)/'adapter_model.safetensors'),device='cpu')
        assert len(state)==len(saved)==120 and set(state)==set(saved)
        assert all(torch.equal(v.detach().cpu().to(saved[k].dtype),saved[k]) for k,v in state.items())
        for value in model.parameters():value.requires_grad_(False)
        model.eval()
    assert len(model.peft_config)==1
    return model

def predict(model,p,case,language):
    # This helper has no target/label/request argument: statistics come only
    # from the exact current input RGB, outside the text prompt.
    set_features(model,features(ROOT/case['image']))
    return previous.predict(model,p,case,language)

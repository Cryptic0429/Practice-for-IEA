"""Language-only q/v/o and MLP LoRA, preserving original A at initialization."""
import hashlib,json
from pathlib import Path
import torch
from peft import LoraConfig,PeftModel,get_peft_model,prepare_model_for_kbit_training,set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import BitsAndBytesConfig,Qwen2_5_VLForConditionalGeneration
from stage2_runtime import ROOT,BASE,ADAPTER,processor,prompt,image_input,encoded,adapter_state,tensor_digest
START=ROOT/'models/goal-v2-A-expanded-start-v1'
TARGET=r'model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))'
def canonical(k):return k.replace('.model.language_model.','.model.')
def base_model():
    base=Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    return prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
def assert_trainable(model):
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    assert len(names)==336,(len(names),names[:10])
    assert all('lora_' in n and '.default.' in n and 'visual' not in n and
               any(x in n for x in ['.self_attn.q_proj.','.self_attn.v_proj.','.self_attn.o_proj.','.mlp.gate_proj.','.mlp.up_proj.','.mlp.down_proj.']) for n in names)
    return names
def build_start():
    model=get_peft_model(base_model(),LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=TARGET,bias='none',task_type='CAUSAL_LM',init_lora_weights=True))
    old={canonical(k):v for k,v in load_file(str(ADAPTER/'adapter_model.safetensors')).items()}
    state=adapter_state(model);assert len(state)==336
    copied=[]
    for k,v in state.items():
        if canonical(k) in old:state[k]=old[canonical(k)];copied.append(canonical(k))
        elif 'lora_B' in k:assert torch.count_nonzero(v)==0
    assert set(copied)==set(old) and len(copied)==112
    set_peft_model_state_dict(model,state,adapter_name='default')
    loaded=adapter_state(model)
    assert all(torch.equal(loaded[k],old[canonical(k)]) for k in loaded if canonical(k) in old)
    assert_trainable(model);return model
def load_model(adapter=START,trainable=True):
    if Path(adapter)==START:
        report=json.loads((START/'start-verification.json').read_text());assert report['passed']
        assert hashlib.sha256((START/'adapter_model.safetensors').read_bytes()).hexdigest()==report['adapter_sha256']
    model=PeftModel.from_pretrained(base_model(),adapter,is_trainable=trainable)
    saved={canonical(k):v for k,v in load_file(str(Path(adapter)/'adapter_model.safetensors')).items()}
    current={canonical(k):v for k,v in adapter_state(model).items()}
    assert len(saved)==336 and current.keys()==saved.keys()
    assert all(torch.equal(current[k],v) for k,v in saved.items())
    if trainable:assert_trainable(model)
    return model

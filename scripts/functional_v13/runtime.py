"""Matched data/capacity hypothesis: preserve R11 prompt, editor and parent.

One adapter per model. The broader route adds language k/o and MLP projections
with zero B, while copying all 116 trained parent tensors, including merger B.
"""
from pathlib import Path
import torch
from functional_v1.common import ROOT,BASE,read,sha
from functional_v1.runtime import processor,encode,generate,verify_adapter
from functional_v7.runtime import canonical,MERGER_PATTERN
from functional_v9.runtime import verify_origin as verify_original_origin
from functional_v11.runtime import build_batch
from functional_v11.outcomes import input_row,parse

EXP=ROOT/'experiments/functional-v13-20261106'
DATA=ROOT/'data/training/functional-v13-20261106'
PARENT=ROOT/'experiments/functional-v11-20261106/runs/language_merger-seed-20261123/final'
PARENT_SHA='12243ea5f08ab9e2e75b082fd6e6a1e6e8b1efbd7eb4e102be3563ba084c4a56'
ROUTES=('language_merger','all_language_merger')
WIDE_PATTERN=r'.*(?:language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))|visual\.merger\.mlp\.(?:0|2))$'


def expected_count(route):
 assert route in ROUTES
 return 116 if route=='language_merger' else 396


def verify_origin():
 identity=verify_original_origin('base')
 assert sha(PARENT/'adapter_model.safetensors')==PARENT_SHA
 config=read(PARENT/'adapter_config.json')
 assert Path(config['base_model_name_or_path']).resolve()==BASE.resolve()
 assert (config['r'],config['lora_alpha'],config['lora_dropout'])==(8,16,0)
 return identity


def create_adapter(base,route):
 from peft import LoraConfig,get_peft_model
 assert route in ROUTES
 return get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,bias='none',task_type='CAUSAL_LM',
  target_modules=MERGER_PATTERN if route=='language_merger' else WIDE_PATTERN))


def transfer_parent(model,route):
 from peft import get_peft_model_state_dict,set_peft_model_state_dict
 from safetensors.torch import load_file
 saved=load_file(str(PARENT/'adapter_model.safetensors'),device='cpu')
 expected={canonical(k):v for k,v in saved.items()};actual=get_peft_model_state_dict(model)
 assert len(saved)==len(expected)==116 and len(actual)==expected_count(route)
 assert sum('visual' in k for k in expected)==4
 assert len({canonical(k) for k in actual})==len(actual)
 assert set(expected)<={canonical(k) for k in actual}
 added={k:v.detach().cpu().clone() for k,v in actual.items() if canonical(k) not in expected}
 assert len(added)==(0 if route=='language_merger' else 280)
 assert all('visual' not in k for k in added)
 assert all(torch.count_nonzero(v)==0 for k,v in added.items() if 'lora_B' in k)
 result=set_peft_model_state_dict(model,{k:expected[canonical(k)].to(v.dtype) if canonical(k) in expected else v for k,v in actual.items()})
 assert not result.unexpected_keys
 after=get_peft_model_state_dict(model)
 assert all(torch.equal(after[k].detach().cpu().to(expected[canonical(k)].dtype),expected[canonical(k)]) for k in after if canonical(k) in expected)
 assert all(torch.equal(after[k].detach().cpu(),v) for k,v in added.items())
 assert set(model.peft_config)=={'default'}
 return dict(route=route,parent_tensors_copied_exactly=116,language_parent_tensors_copied_exactly=112,
  visual_parent_tensors_copied_exactly=4,added_language_tensors=len(added),added_B_zero=True,
  inherited_visual_B_preserved=True,adapter_count=1,stacked_adapters=False)


def set_trainable(model,route):
 names=[]
 for name,value in model.named_parameters():
  active='lora_' in name;value.requires_grad_(active)
  if active:
   if 'visual' in name:assert '.merger.mlp.' in name
   elif route=='language_merger':assert '.q_proj.' in name or '.v_proj.' in name
   else:assert any('.'+key+'.' in name for key in ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'))
   value.data=value.data.float();names.append(name)
 assert len(names)==expected_count(route)
 model.config.use_cache=False;model.train()
 return names


def load_model(route,adapter=None,train=False,seed=None):
 from transformers import BitsAndBytesConfig,Qwen2_5_VLForConditionalGeneration,set_seed
 from peft import PeftModel,prepare_model_for_kbit_training
 assert route in ROUTES and (not train or (adapter is None and seed is not None))
 verify_origin()
 base,information=Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE,torch_dtype=torch.bfloat16,
  attn_implementation='sdpa',device_map={'':0},output_loading_info=True,
  quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
 assert not any(information.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),information
 if train:
  base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
  set_seed(seed);model=create_adapter(base,route);transfer=transfer_parent(model,route);set_trainable(model,route)
  model.functional_initialization=dict(origin='base',parent_path=str(PARENT.relative_to(ROOT)),parent_sha256=PARENT_SHA,
   full_weight_loading_information=information,**transfer)
 else:
  assert adapter is not None
  config=read(Path(adapter)/'adapter_config.json')
  assert Path(config['base_model_name_or_path']).resolve()==BASE.resolve()
  assert config['target_modules']==(MERGER_PATTERN if route=='language_merger' else WIDE_PATTERN)
  model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False);verify_adapter(model,adapter)
  from peft import get_peft_model_state_dict
  assert len(get_peft_model_state_dict(model))==expected_count(route)
  for weight in model.parameters():weight.requires_grad_(False)
  model.eval()
 return model


def predict(model,p,case,language):
 response=generate(model,p,encode(p,input_row(case,language)),max_tokens=256)
 return response,parse(response)

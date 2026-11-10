"""CPU/meta audit: no forward, no GPU, no capability claim."""
import gc,os
from pathlib import Path
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig,Qwen2_5_VLForConditionalGeneration
from functional_v1.common import BASE,write,sha,stamp
from .runtime import EXP,PARENT,PARENT_SHA,ROUTES,create_adapter,transfer_parent,set_trainable,expected_count


def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 target=EXP/'runtime-transfer-audit.json';assert not target.exists()
 assert sha(PARENT/'adapter_model.safetensors')==PARENT_SHA
 config=AutoConfig.from_pretrained(BASE);results=[]
 for route in ROUTES:
  with init_empty_weights():
   base=Qwen2_5_VLForConditionalGeneration(config);model=create_adapter(base,route)
  for name,module in model.named_modules():
   if isinstance(module,torch.nn.Linear) and ('.lora_A.' in name or '.lora_B.' in name):
    module.to_empty(device='cpu')
    torch.nn.init.zeros_(module.weight) if '.lora_B.' in name else torch.nn.init.kaiming_uniform_(module.weight,a=5**.5)
  receipt=transfer_parent(model,route);names=set_trainable(model,route)
  assert len(names)==expected_count(route)
  assert all(v.is_meta for k,v in model.named_parameters() if 'lora_' not in k)
  assert all(not v.is_meta and v.dtype==torch.float32 for k,v in model.named_parameters() if k in names)
  receipt.update(trainable_tensors=len(names),trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
   all_base_parameters_meta=True,forward_or_gradient_test_performed=False)
  results.append(receipt);del model,base;gc.collect()
 write(target,dict(at=stamp(),status='CPU_meta_exact_116_parent_transfer_and_new_zero_B_verified',parent_sha256=PARENT_SHA,
  runtime_sha256=sha(Path(__file__).with_name('runtime.py')),results=results,GPU_hours=0,new_training_started=False,
  limitation='No quantized GPU load, initial-function parity, finite-gradient or model capability proof.'))
 print('CPU/meta adapter layouts verified',[(r['route'],r['trainable_tensors']) for r in results])


if __name__=='__main__':main()

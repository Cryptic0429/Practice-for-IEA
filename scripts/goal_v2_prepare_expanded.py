"""Verify zero-added expanded adapters match A logits before any training."""
import gc,hashlib,json,os,subprocess
from pathlib import Path
import torch
from transformers import set_seed
from trl.models.utils import peft_module_casting_to_bf16
from stage2_runtime import ROOT,ADAPTER,load_model as original_model,processor,encoded
from goal_v2_json_runtime import prepare_row
from goal_v2_expanded_runtime import START,build_start,adapter_state
def main():
    gpu=os.environ['CUDA_VISIBLE_DEVICES'];assert gpu=='3'
    assert int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))<512
    START.mkdir(exist_ok=False);set_seed(20261103);torch.set_num_threads(4)
    rows=json.loads((ROOT/'data/goal-v2-20261103/train-memorize64.json').read_text())[:2];p=processor();expected=[]
    old=original_model();peft_module_casting_to_bf16(old);old.eval()
    for row in rows:
        inputs={k:v.to('cuda') for k,v in encoded(p,prepare_row(row)).items()}
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):logits=old(**inputs,use_cache=False).logits[:,-1].float().cpu()
        expected.append(logits);del inputs,logits
    del old;gc.collect();torch.cuda.empty_cache()
    model=build_start();peft_module_casting_to_bf16(model);model.eval();differences=[]
    for row,reference in zip(rows,expected):
        inputs={k:v.to('cuda') for k,v in encoded(p,prepare_row(row)).items()}
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):logits=model(**inputs,use_cache=False).logits[:,-1].float().cpu()
        difference=float((logits-reference).abs().max());differences.append(difference)
        assert torch.equal(logits,reference),difference
    model.save_pretrained(START)
    weights=adapter_state(model);assert len(weights)==336
    report={'passed':True,'adapter_tensors':336,'trainable_parameters':sum(v.numel() for v in weights.values()),'rank':8,'alpha':16,
            'copied_qv_tensors':112,'new_module_lora_B_initialization':'zero','max_logit_differences':differences,'probe_ids':[r['id'] for r in rows],
            'adapter_sha256':hashlib.sha256((START/'adapter_model.safetensors').read_bytes()).hexdigest(),
            'original_A_sha256':hashlib.sha256((ADAPTER/'adapter_model.safetensors').read_bytes()).hexdigest(),
            'optimizer_updates':0,'scope':'Two training-input next-token logits exactly equal to original A; no held-out inputs.'}
    (START/'start-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

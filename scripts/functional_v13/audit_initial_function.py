"""Quantized GPU parity on training-pool prefixes before any optimization.

Both layouts use the SAME k-bit preparation, evaluation mode and processor.
No development/acceptance images, target labels, or optimizer steps are used.
"""
import argparse,gc,hashlib,os
import torch
from functional_v1.common import ROOT,read,write,sha,stamp,GPUJob
from functional_v1.runtime import encode,generate
from functional_v11.outcomes import input_row
from .runtime import EXP,ROUTES,load_model,processor,PARENT_SHA


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',required=True,choices=['0','2','3']);args=ap.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==args.gpu
 target=EXP/'initial-function-parity.json';assert not target.exists()
 cpu=read(EXP/'runtime-transfer-audit.json');assert cpu['parent_sha256']==PARENT_SHA
 source=read(EXP/'primary-candidates/manifest.json');bank={i['id']:i for i in source['items']}
 ids=['train-native-078','train-native-144','train-extension-396'];cases=[]
 for sid in ids:
  i=bank[sid];assert sha(ROOT/i['input_image'])==i['input_sha256']
  cases.append(dict(id=sid,image=i['input_image'],instructions=i['instructions']))
 p=processor();batches=[encode(p,input_row(c,'en')) for c in cases];records=[];baseline=[]
 with GPUJob(args.gpu,'v13-training-prefix-initial-function-parity',max_seconds=600) as budget:
  torch.set_num_threads(4)
  for route in ROUTES:
   model=load_model(route,train=True,seed=20261127);model.eval()
   assert not model.training
   for index,(case,batch) in enumerate(zip(cases,batches)):
    moved={k:v.to('cuda') for k,v in batch.items()}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
     result=model(**moved,use_cache=False);logits=result.logits[:,-1].float().cpu()
    assert torch.isfinite(logits).all();del result,moved
    response=generate(model,p,batch,max_tokens=256)
    if route==ROUTES[0]:baseline.append((logits.clone(),response));difference=0.
    else:
     old,text=baseline[index];difference=float((old-logits).abs().max())
     assert torch.equal(old,logits),(case['id'],difference)
     assert response==text,(case['id'],'initial greedy response changed')
    records.append(dict(route=route,source_id=case['id'],language='en',response=response,last_prefix_logits_sha256=hashlib.sha256(logits.numpy().tobytes()).hexdigest(),max_difference_from_thin=difference,
     parent_tensors_copied_exactly=model.functional_initialization['parent_tensors_copied_exactly']))
    budget.tick()
   del model;gc.collect();torch.cuda.empty_cache()
 write(target,dict(at=stamp(),status='passed',parent_sha256=PARENT_SHA,records=records,
  both_prepared_kbit_and_eval=True,image_max_pixels=262144,optimizer_steps=0,training_targets_used=False,
  development_or_acceptance_inputs_used=False,limitation='Three training-pool English prefixes only. Exact initial parity does not predict training quality or generalization.'))
 print('Quantized initial parity passed on3 training-only inputs; no optimization or heldout use.')


if __name__=='__main__':main()

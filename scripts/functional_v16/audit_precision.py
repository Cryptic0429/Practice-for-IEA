"""Audit training-prepared versus adapter-reloaded inference; training prefixes only."""
import argparse,gc,hashlib,os
from collections import Counter
import torch
from functional_v1.common import ROOT,read,write,sha,stamp,GPUJob
from functional_v1.runtime import encode,generate
from functional_v11.outcomes import input_row
from functional_v13.runtime import PARENT,PARENT_SHA,load_model,processor

EXP=ROOT/'experiments/functional-v16-20261106'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',required=True,choices=['0','2','3']);args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==args.gpu
    target=EXP/'training-prefix-precision-audit.json';assert not target.exists()
    bank={c['id']:c for c in read(ROOT/'experiments/functional-v13-20261106/primary-candidates/manifest.json')['items']}
    cases=[]
    for sid in ('train-native-078','train-native-144','train-extension-396'):
        c=bank[sid];assert sha(ROOT/c['input_image'])==c['input_sha256']
        cases.append(dict(id=sid,image=c['input_image'],instructions=c['instructions']))
    p=processor();batches=[encode(p,input_row(c,'en')) for c in cases];baseline=[];records=[];dtypes={}
    with GPUJob(args.gpu,'v16-training-prefix-precision-audit',max_seconds=600) as budget:
        torch.set_num_threads(4)
        for variant in ('adapter_reloaded_inference','training_prepared_eval'):
            model=load_model('language_merger',adapter=PARENT) if variant=='adapter_reloaded_inference' else load_model('language_merger',train=True,seed=20261127)
            model.eval()
            named={n:str(v.dtype) for n,v in model.named_parameters() if 'lora_' not in n and any(n.endswith(s) for s in ('patch_embed.proj.weight','embed_tokens.weight','lm_head.weight','layers.0.input_layernorm.weight','language_model.norm.weight'))}
            dtypes[variant]=dict(selected=named,all_parameter_dtypes=dict(Counter(str(v.dtype) for v in model.parameters())))
            for index,(case,batch) in enumerate(zip(cases,batches)):
                moved={k:v.to('cuda') for k,v in batch.items()}
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    result=model(**moved,use_cache=False);logits=result.logits[:,-1].float().cpu()
                assert torch.isfinite(logits).all();del result,moved
                response=generate(model,p,batch,max_tokens=256)
                if not index and variant=='adapter_reloaded_inference':assert sha(PARENT/'adapter_model.safetensors')==PARENT_SHA
                if variant=='adapter_reloaded_inference':baseline.append((logits,response));difference=0.;same_response=True;same_logits=True
                else:
                    previous,text=baseline[index];difference=float((previous-logits).abs().max());same_logits=torch.equal(previous,logits);same_response=text==response
                records.append(dict(variant=variant,case_id=case['id'],response=response,last_logits_sha256=hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
                    max_last_logit_difference=difference,exact_logits_identical_to_reload=same_logits,greedy_response_identical_to_reload=same_response))
                budget.tick()
            del model;gc.collect();torch.cuda.empty_cache()
    write(target,dict(at=stamp(),status='audit_completed_not_a_functional_result',parent_sha256=PARENT_SHA,
        variants=dtypes,records=records,optimizer_steps=0,model_parameters_changed=False,training_prefixes_only=True,
        development_or_acceptance_inputs_used=False,round14_runtime_changed=False,
        limitation='3 training-pool English prefixes and one parent checkpoint only. Dtype/logit/response differences are implementation diagnostics, not evidence that either variant edits better. No policy training, new labels or image quality grades.'))
    print('Precision audit complete; prepared greedy matches:',sum(r['greedy_response_identical_to_reload'] for r in records if r['variant']=='training_prepared_eval'),'/3')


if __name__=='__main__':main()

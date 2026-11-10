"""Quantized parent parity on three training-only prefixes, with zero optimizer steps."""
import argparse,gc,hashlib,os
import torch
from functional_v1.common import ROOT,read,write,sha,stamp,GPUJob
from functional_v1.runtime import encode,generate
from functional_v11.outcomes import input_row,EDITOR_VERSION
from functional_v13.runtime import load_model as load_control
from .runtime import EXP,load_model,processor,PARENT_SHA


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',required=True,choices=['0','2','3']);args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==args.gpu
    target=EXP/'initial-function-parity.json';assert not target.exists()
    assert read(EXP/'runtime-transfer-audit.json')['parent_sha256']==PARENT_SHA
    bank={i['id']:i for i in read(ROOT/'experiments/functional-v13-20261106/primary-candidates/manifest.json')['items']}
    cases=[]
    for sid in ['train-native-078','train-native-144','train-extension-396']:
        item=bank[sid];assert sha(ROOT/item['input_image'])==item['input_sha256']
        cases.append(dict(id=sid,image=item['input_image'],instructions=item['instructions']))
    p=processor();batches=[encode(p,input_row(c,'en')) for c in cases];baseline=[];records=[]
    with GPUJob(args.gpu,'v14-training-prefix-initial-function-parity',max_seconds=600) as budget:
        with budget.ledger() as ledger:
            next(j for j in ledger['jobs'] if j['id']==budget.id).update(editor=EDITOR_VERSION,runtime='functional_v14.runtime')
        torch.set_num_threads(4)
        for route in ('matched_language_merger','visual_late4'):
            model=load_control('language_merger',train=True,seed=20261127) if route=='matched_language_merger' else load_model('visual_late4',train=True,seed=20261127)
            model.eval()
            for index,(case,batch) in enumerate(zip(cases,batches)):
                moved={k:v.to('cuda') for k,v in batch.items()}
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    result=model(**moved,use_cache=False);logits=result.logits[:,-1].float().cpu()
                assert torch.isfinite(logits).all();del result,moved
                response=generate(model,p,batch,max_tokens=256)
                if route=='matched_language_merger':baseline.append((logits.clone(),response));delta=0.
                else:
                    old,text=baseline[index];delta=float((old-logits).abs().max())
                    assert torch.equal(old,logits),(case['id'],delta)
                    assert response==text
                records.append(dict(route=route,source_id=case['id'],language='en',response=response,max_difference_from_control=delta,
                    last_prefix_logits_sha256=hashlib.sha256(logits.numpy().tobytes()).hexdigest(),parent_tensors_copied_exactly=116))
                budget.tick()
            del model;gc.collect();torch.cuda.empty_cache()
    write(target,dict(at=stamp(),status='passed',parent_sha256=PARENT_SHA,records=records,both_prepared_kbit_and_eval=True,
        image_max_pixels=262144,optimizer_steps=0,training_targets_used=False,development_or_acceptance_inputs_used=False,
        limitation='Three training-only English prefixes; exact initial function parity is implementation evidence, not editing-quality or generalization evidence.'))
    print('Initial quantized parity passed on3 training-only prefixes, zero optimization.')


if __name__=='__main__':main()

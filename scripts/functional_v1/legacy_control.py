"""Historical offline prompt/parser/editor control, without external translation."""
import argparse
import json
import sys
from pathlib import Path
from .common import ROOT, DATA, EXP, EVAL, PARENT, GPUJob, read, write, append, sha, stamp


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gpu',choices=['0','2','3'],default='2')
    args=ap.parse_args()
    sys.path.insert(0,str(ROOT/'repo'))
    import cv2
    import torch
    from transformers import set_seed
    from trl.models.utils import peft_module_casting_to_bf16
    from stage2_runtime import load_model,processor,encoded
    from goal_v2_json_runtime import prepare_row
    from evaluation_core import parse_parameters,sample_seed
    from src.image_editor import ImageEditor
    cases=read(DATA/'development-cases.json')
    original=read(ROOT/'data/training/quality-v2-20261103/train3k.json')[0]
    protocol=read(ROOT/'configs/evaluation-protocol-v1.json')
    mid='current-legacy'
    output=EVAL/'development'/mid
    output.mkdir(parents=True,exist_ok=True)
    identity={'model_id':mid,'adapter':str(PARENT),'adapter_sha256':sha(PARENT/'adapter_model.safetensors'),
        'case_sha256':sha(DATA/'development-cases.json'),'editor':'historical_upstream_editor',
        'prompt':'Historical ImageEdit conversations plus goal_v2_json_runtime.SCHEMA',
        'runtime':'stage2_runtime + original final inference casting and deterministic seed',
        'scope':'Historical offline path; no external API translation. Bilingual request directly inserted.',
        'code_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/stage2_runtime.py',ROOT/'scripts/goal_v2_json_runtime.py',ROOT/'scripts/evaluation_core.py',ROOT/'repo/src/image_editor.py']}}
    if (output/'identity.json').exists():assert read(output/'identity.json')==identity
    else:write(output/'identity.json',identity)
    path=output/'predictions.jsonl'
    done={(r['case_id'],r['language']) for r in map(json.loads,path.read_text().splitlines())} if path.exists() else set()
    with GPUJob(args.gpu,'legacy-offline-development-control',max_seconds=3600) as budget:
        torch.set_num_threads(4)
        model=load_model(PARENT,trainable=True)
        peft_module_casting_to_bf16(model)
        for weight in model.parameters():weight.requires_grad_(False)
        model.eval();p=processor()
        for case in cases:
            for language in ('zh','en'):
                if (case['id'],language) in done:continue
                budget.tick()
                row={'images':[str(ROOT/case['image'])], 'conversations':[dict(m) for m in original['conversations'][:2]]}
                row['conversations'][1]['content']='<image> You can follow user instructions to determine the adjustments:\n'+case['instructions'][language]+'\nFirst, provide your thoughts about how to optimize the input image. Then, output the parameters in a JSON object format.'
                row=prepare_row(row)
                rec={'case_id':case['id'],'group_id':case['group_id'],'family':case['family'],'action':case['action'],
                     'language':language,'model_id':mid,'input_image':case['image'],'input_sha256':case['input_sha256'],
                     'instruction':case['instructions'][language],'status':'failed'}
                try:
                    set_seed(sample_seed(case['id']+'-'+language,20261101))
                    inputs={k:v.to('cuda') for k,v in encoded(p,row).items()}
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        generated=model.generate(**inputs,do_sample=False,max_new_tokens=512,repetition_penalty=1.,
                            pad_token_id=p.tokenizer.pad_token_id,eos_token_id=p.tokenizer.eos_token_id,use_cache=True,disable_compile=True)
                    tokens=generated[0,inputs['input_ids'].shape[1]:].tolist()
                    if len(tokens)>=512 and p.tokenizer.eos_token_id not in tokens:raise ValueError('Legacy generation truncated')
                    response=p.decode(tokens,skip_special_tokens=True)
                    parsed=parse_parameters(response,protocol)
                    if not parsed['valid']:raise ValueError(parsed['error'])
                    image=ImageEditor(str(ROOT/case['image'])).process_image(parsed['parameters'])
                    target=output/'images'/f"{case['id']}-{language}.png"
                    target.parent.mkdir(exist_ok=True)
                    assert cv2.imwrite(str(target),image)
                    rec.update(status='executed',response=response,parameters=parsed['parameters'],output_image=str(target.relative_to(ROOT)),output_sha256=sha(target))
                except Exception as error:rec['error']=f'{type(error).__name__}: {error}'
                append(path,dict(rec,completed_at=stamp()))
                done.add((case['id'],language))
                write(output/'status.json',{'status':'running','completed':len(done),'total':100})
        write(output/'status.json',{'status':'completed_predictions','completed':len(done),'total':100})


if __name__=='__main__':main()

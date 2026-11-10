"""Compare only image budget on four fixed TRAIN prefixes, without optimization."""
import argparse
import hashlib
import json
import math
import os
from PIL import Image,ImageOps
from functional_v1.common import ROOT,read,write,append,sha,stamp,GPUJob
from functional_v1.runtime import encode,messages,generate
from functional_v13.runtime import PARENT,PARENT_SHA,load_model,processor,verify_origin
from functional_v15.outcomes import input_row,parse,execute,EDITOR_VERSION,system_prompt
from functional_v15.editor import load_rgb,save

EXP=ROOT/'experiments/functional-v17-20261107'
DATA=ROOT/'data/training/functional-v15-20261106'
IDS=('native-train-n127','r13-train-native-073','native-train-x250','native-train-x151')


def larger_input(path,cap):
    with Image.open(path) as im:
        result=ImageOps.exif_transpose(im).convert('RGB');w,h=result.size
        if w*h>cap:
            factor=math.sqrt(cap/(w*h));result=result.resize((max(28,int(w*factor)),max(28,int(h*factor))))
        if min(result.size)<28:result=result.resize((max(28,result.width),max(28,result.height)))
        return result


def high_encode(p,row):
    msg=messages(row)
    return p(text=[p.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)],
        images=[larger_input(row['images'][0],1048576)],return_tensors='pt',add_special_tokens=False)


def freeze():
    dest=EXP/'protocol.json';assert not dest.exists()
    p15=read(ROOT/'experiments/functional-v15-20261106/training-protocol.json')
    assert all(sha(ROOT/f)==h for f,h in p15['code_hashes'].items())
    cases={c['id']:c for c in read(DATA/'cases.json')}
    selected=[]
    for cid in IDS:
        c=cases[cid];assert sha(ROOT/c['image'])==c['input_sha256']
        selected.append(dict(id=cid,image=c['image'],input_sha256=c['input_sha256'],domain=c['domain'],
            instructions=c['instructions'],expected_change=c.get('expected_change'),must_preserve=c.get('must_preserve')))
    assert [c['domain'] for c in selected].count('portrait')==2
    files=dict(p15['code_hashes']);files[str(__import__('pathlib').Path(__file__).relative_to(ROOT))]=sha(__file__)
    write(dest,dict(at=stamp(),status='frozen_before_training_prefix_resolution_forwards',cases=selected,
        cases_sha256=sha(DATA/'cases.json'),parent_sha256=PARENT_SHA,origin_identity=verify_origin(),
        code_hashes=files,image_max_pixels=[262144,1048576],language='en',
        prompt_sha256=hashlib.sha256(system_prompt().encode()).hexdigest(),editor=EDITOR_VERSION,
        GPU_seconds_cap=600,optimizer_steps=0,development_or_acceptance_used=False,
        hypothesis='Input resizing may hide noise/texture or alter tonal interpretation. Four reused TRAIN sources only; no reference/label/criterion enters model inputs.',
        decision_rule='Different logits/parameters do not establish better editing. Execute decisions and visually assess only actual changed results. No automatic new training, no data admission, no benchmark gain claim.'))
    print('Frozen4 TRAIN photographs(P2,L2), same parent/prompt,262144 versus1048576 pixels.')


def run(gpu):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==gpu
    protocol=read(EXP/'protocol.json');assert not (EXP/'result.json').exists()
    assert all(sha(ROOT/f)==h for f,h in protocol['code_hashes'].items())
    import torch
    torch.set_num_threads(4)
    processors={cap:processor() for cap in protocol['image_max_pixels']}
    processors[1048576].image_processor.max_pixels=1048576
    batches={}
    for case in protocol['cases']:
        row=input_row(case,'en')
        for cap,p in processors.items():
            batch=encode(p,row) if cap==262144 else high_encode(p,row)
            assert batch['input_ids'].shape[1]<2048
            batches[case['id'],cap]=batch
    records=[]
    with GPUJob(gpu,'v17-TRAIN-resolution-diagnostic',max_seconds=600) as budget:
        with budget.ledger() as ledger:
            next(j for j in ledger['jobs'] if j['id']==budget.id).update(editor=EDITOR_VERSION,runtime='exact_functional_v13_parent_loader_resolution_diagnostic_only')
        model=load_model('language_merger',adapter=PARENT)
        for case in protocol['cases']:
            for cap,p in processors.items():
                budget.tick();batch=batches[case['id'],cap]
                row=dict(case_id=case['id'],domain=case['domain'],image_max_pixels=cap,
                    input_tokens=int(batch['input_ids'].shape[1]),image_grid_thw=batch['image_grid_thw'].tolist(),
                    pixel_values_shape=list(batch['pixel_values'].shape),input_sha256=case['input_sha256'],status='failed',
                    output_image=None,output_sha256=None)
                response=generate(model,p,batch,max_tokens=256);row['response']=response
                try:
                    decision=parse(response);result=execute(load_rgb(ROOT/case['image']),decision)
                    row.update(decision=decision,status=result['status'],parameters=result['parameters'])
                    if result['result'] is not None:
                        path=EXP/'outputs'/f'{case["id"]}-{cap}.png';save(path,result['result'])
                        row.update(output_image=str(path.relative_to(ROOT)),output_sha256=sha(path))
                    else:assert result['status']=='declined'
                except (ValueError,TypeError) as e:row['error']=f'{type(e).__name__}: {e}'
                append(EXP/'predictions.jsonl',row);records.append(row)
    pairs=[]
    for cid in IDS:
        low,high=[next(r for r in records if r['case_id']==cid and r['image_max_pixels']==cap) for cap in (262144,1048576)]
        pairs.append(dict(case_id=cid,response_identical=low['response']==high['response'],
            decision_identical=low.get('decision')==high.get('decision'),output_identical=low['output_sha256']==high['output_sha256'],
            low_status=low['status'],high_status=high['status'],input_tokens=[low['input_tokens'],high['input_tokens']]))
    write(EXP/'result.json',dict(at=stamp(),status='diagnostic_completed_no_policy_training',records=records,pairs=pairs,
        protocol_sha256=sha(EXP/'protocol.json'),optimizer_steps=0,parent_sha256=PARENT_SHA,
        development_or_acceptance_used=False,training_data_admission=False,practical_quality_conclusion=None,
        limitation='Four already reviewed TRAIN photographs and one frozen failed parent only. Changed responses/pixels need direct visual review; no generalization, training gain, reliable refusal or user approval claim.'))
    print(json.dumps(pairs))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--gpu',choices=['0','2','3']);a=p.parse_args()
    freeze() if a.freeze else run(a.gpu)

"""Run the exact R29 output head and a separate operative-clause guard."""
import torch
from torch.nn.functional import linear
from transformers import AutoTokenizer,Qwen2_5_VLForConditionalGeneration

from functional_v1.common import ROOT,read,sha,stamp,write
from functional_v27.qwen_text import BASE
from .constants import EXP,PLAN
from .route_guard import explicit_route

R29 = ROOT/'experiments/functional-v29-20261107'


def main():
    plan = read(PLAN)
    proto,head_path,result = R29/'protocol.json',R29/'run/heads.pt',R29/'result.json'
    for path,key in ((proto,'prior_r29_protocol_sha256'),(head_path,'prior_r29_heads_sha256'),
                     (result,'prior_r29_result_sha256')):
        if sha(path)!=plan[key]:
            raise ValueError('R29 head/protocol changed')
    parent = read(proto)
    examples = [dict(id=c['id'],cohort='held_photo',expected='global_down',requests=c['requests'])
                for c in plan['photos'] if c['cohort']=='held']
    examples += [dict(c,cohort='held_control') for c in plan['controls']]
    rows = [dict(id=c['id'],cohort=c['cohort'],expected=c['expected'],language=l,instruction=s)
            for c in examples for l,s in c['requests'].items()]
    old = [r for r in read(result)['rows'] if r['id'] in ('p021-global_down','held-text-11')]
    rows += [dict(id=r['id'],cohort='r29_regression',expected=r['expected'],
                  language=r['language'],instruction=r['instruction']) for r in old]
    torch.set_num_threads(plan['cpu_threads'])
    torch.set_num_interop_threads(1)
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    tokenizer.padding_side='right'
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE,torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).eval()
    if any(p.device.type!='cpu' for p in model.parameters()):
        raise ValueError('CPU-only path violated')
    w=torch.load(head_path,map_location='cpu',weights_only=True)['augmented']
    choices=list(parent['labels'].values())
    with torch.inference_mode():
        for start in range(0,len(rows),8):
            batch=rows[start:start+8]
            prompts=[tokenizer.apply_chat_template([
                dict(role='system',content=parent['system_prompt']),dict(role='user',content=r['instruction'])
            ],tokenize=False,add_generation_prompt=True) for r in batch]
            tokens=tokenizer(prompts,padding=True,add_special_tokens=False,return_tensors='pt')
            ids,mask=tokens.input_ids,tokens.attention_mask
            pos,_=model.get_rope_index(ids,None,None,mask)
            hidden=model.model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False).last_hidden_state
            final=hidden[torch.arange(len(batch)),mask.sum(dim=1)-1]
            normalized=((final-w['mean'])/w['scale']).to(w['weight'].dtype)
            for r,score in zip(batch,linear(normalized,w['weight'],w['bias'])):
                selected=choices[int(score.argmax())]
                guard=explicit_route(r['instruction'])
                r.update(head=selected,explicit=guard,guarded=selected if selected==guard else 'defer',
                         logits={k:float(v) for k,v in zip(choices,score)})
            print(f'Routed {start+len(batch)}/{len(rows)}',flush=True)
    known={(r['id'],r['language']):r['predictions']['augmented_head']['route'] for r in old}
    if any(r['head']!=known[(r['id'],r['language'])] for r in rows if r['cohort']=='r29_regression'):
        raise ValueError('Full head replay did not reproduce previous routes')
    write(EXP/'route-result.json',dict(at=stamp(),protocol_sha256=sha(PLAN),guard_code_sha256=sha(
        ROOT/'scripts/functional_v30/route_guard.py'),rows=rows,GPU_hours=0))


if __name__=='__main__':
    main()

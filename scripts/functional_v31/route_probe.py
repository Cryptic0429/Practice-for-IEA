"""Test new literal requests with the frozen R29 head and the exact R30 agreement rule."""
import torch
from torch.nn.functional import linear
from transformers import AutoTokenizer,Qwen2_5_VLForConditionalGeneration

from functional_v1.common import ROOT,read,sha,stamp,write
from functional_v27.qwen_text import BASE
from functional_v30.route_guard import explicit_route
from .render import EXP,PLAN,R30


def main():
    plan=read(PLAN)
    base_protocol=R30/'protocol.json'
    if sha(base_protocol)!=plan['r30_protocol_sha256']:
        raise ValueError('R30 protocol changed')
    r30=read(base_protocol)
    r29=ROOT/'experiments/functional-v29-20261107'
    prompt,weight=r29/'protocol.json',r29/'run/heads.pt'
    if sha(prompt)!=r30['prior_r29_protocol_sha256'] or sha(weight)!=r30['prior_r29_heads_sha256']:
        raise ValueError('Frozen R29 head changed')
    parent=read(prompt)
    routes=list(parent['labels'].values())
    rows=[dict(id=c['id'],language=l,instruction=s) for c in plan['photos'] if c['cohort']=='held'
          for l,s in c['requests'].items()]
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    tok.padding_side='right'
    base=Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE,torch_dtype=torch.float32,
        attn_implementation='sdpa',local_files_only=True).eval()
    if any(p.device.type!='cpu' for p in base.parameters()):
        raise ValueError('CPU-only probe violated')
    w=torch.load(weight,map_location='cpu',weights_only=True)['augmented']
    with torch.inference_mode():
        for i in range(0,len(rows),8):
            group=rows[i:i+8]
            prompts=[tok.apply_chat_template([dict(role='system',content=parent['system_prompt']),
                dict(role='user',content=r['instruction'])],tokenize=False,add_generation_prompt=True) for r in group]
            tokens=tok(prompts,padding=True,add_special_tokens=False,return_tensors='pt')
            ids,mask=tokens.input_ids,tokens.attention_mask
            pos,_=base.get_rope_index(ids,None,None,mask)
            hidden=base.model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False).last_hidden_state
            last=hidden[torch.arange(len(group)),mask.sum(1)-1]
            z=((last-w['mean'])/w['scale']).to(w['weight'].dtype)
            for r,v in zip(group,linear(z,w['weight'],w['bias'])):
                selected=routes[int(v.argmax())]
                explicit=explicit_route(r['instruction'])
                r.update(head=selected,explicit=explicit,guarded=selected if selected==explicit else 'defer',
                         logits=dict(zip(routes,map(float,v))))
            print('Routed',i+len(group),len(rows),flush=True)
    write(EXP/'route-result.json',dict(at=stamp(),protocol_sha256=sha(PLAN),head_sha256=sha(weight),
        guard_sha256=sha(ROOT/'scripts/functional_v30/route_guard.py'),rows=rows,GPU_hours=0))


if __name__=='__main__':
    main()

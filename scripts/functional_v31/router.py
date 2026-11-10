"""The frozen R29 output head, reusable by the R31 CPU probe and editor."""
import torch
from torch.nn.functional import linear
from transformers import AutoTokenizer,Qwen2_5_VLForConditionalGeneration

from functional_v1.common import ROOT,read,sha
from functional_v27.qwen_text import BASE
from functional_v30.route_guard import explicit_route
from .render import PLAN,R30


class Router:
    def __init__(self):
        plan=read(PLAN)
        base_protocol=R30/'protocol.json'
        if sha(base_protocol)!=plan['r30_protocol_sha256']:
            raise ValueError('R30 protocol changed')
        r30=read(base_protocol)
        r29=ROOT/'experiments/functional-v29-20261107'
        prompt,self.weight_path=r29/'protocol.json',r29/'run/heads.pt'
        if sha(prompt)!=r30['prior_r29_protocol_sha256'] or sha(self.weight_path)!=r30['prior_r29_heads_sha256']:
            raise ValueError('Frozen R29 head changed')
        parent=read(prompt)
        self.routes=list(parent['labels'].values())
        self.system=parent['system_prompt']
        torch.set_num_threads(16)
        self.tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
        self.tok.padding_side='right'
        self.base=Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE,torch_dtype=torch.float32,
            attn_implementation='sdpa',local_files_only=True).eval()
        if any(p.device.type!='cpu' for p in self.base.parameters()):
            raise ValueError('CPU-only path violated')
        self.w=torch.load(self.weight_path,map_location='cpu',weights_only=True)['augmented']

    @torch.inference_mode()
    def classify(self,texts):
        prompts=[self.tok.apply_chat_template([dict(role='system',content=self.system),
            dict(role='user',content=text)],tokenize=False,add_generation_prompt=True) for text in texts]
        tokens=self.tok(prompts,padding=True,add_special_tokens=False,return_tensors='pt')
        ids,mask=tokens.input_ids,tokens.attention_mask
        pos,_=self.base.get_rope_index(ids,None,None,mask)
        hidden=self.base.model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False).last_hidden_state
        last=hidden[torch.arange(len(texts)),mask.sum(1)-1]
        z=((last-self.w['mean'])/self.w['scale']).to(self.w['weight'].dtype)
        rows=[]
        for text,v in zip(texts,linear(z,self.w['weight'],self.w['bias'])):
            selected=self.routes[int(v.argmax())]
            explicit=explicit_route(text)
            rows.append(dict(head=selected,explicit=explicit,guarded=selected if selected==explicit else 'defer',
                             logits=dict(zip(self.routes,map(float,v)))))
        return rows

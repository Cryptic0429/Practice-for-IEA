"""Check if factoring scope and direction improves the frozen model's text routing."""
from collections import Counter

import torch
from torch.nn.functional import linear
from transformers import AutoTokenizer,Qwen2_5_VLForConditionalGeneration

from functional_v1.common import read,sha,stamp,write
from functional_v27.qwen_text import BASE
from .boards import EXP

JOINT = dict(zip('ABCDE',('shadow_up','shadow_down','global_up','global_down','defer')))
FACTORED = {('A','A'):'shadow_up',('A','B'):'shadow_down',('B','A'):'global_up',('B','B'):'global_down'}


def main():
    path = EXP/'route-protocol.json'
    plan = read(path)
    torch.set_num_threads(plan['cpu_threads'])
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    tokenizer.padding_side = 'right'
    token_ids = [tokenizer.encode(s,add_special_tokens=False) for s in JOINT]
    if any(len(row)!=1 for row in token_ids):
        raise ValueError('Options must be single tokens')
    token_ids = [v[0] for v in token_ids]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE,torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).eval()
    if any(p.device.type!='cpu' for p in model.parameters()):
        raise ValueError('CPU-only path violated')
    rows = [dict(id=c['id'],source=c['source'],language=lang,instruction=text,expected=c['expected'],votes={})
            for c in plan['examples'] for lang,text in c['instructions'].items()]
    batch_size = plan['batch_size']
    with torch.inference_mode():
        for axis,system in plan['prompts'].items():
            options = 'ABCDE' if axis=='joint' else 'ABC'
            weights = model.lm_head.weight[token_ids[:len(options)]]
            for start in range(0,len(rows),batch_size):
                subset = rows[start:start+batch_size]
                prompts = [tokenizer.apply_chat_template([dict(role='system',content=system),
                    dict(role='user',content=r['instruction'])],tokenize=False,add_generation_prompt=True) for r in subset]
                tokens = tokenizer(prompts,add_special_tokens=False,padding=True,return_tensors='pt')
                ids,mask = tokens.input_ids,tokens.attention_mask
                pos,_ = model.get_rope_index(ids,None,None,mask)
                hidden = model.model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False).last_hidden_state
                scores = linear(hidden[torch.arange(len(subset)),mask.sum(dim=1)-1],weights)
                for r,score in zip(subset,scores):
                    r['votes'][axis] = dict(choice=options[int(score.argmax())],
                        logits={option:float(value) for option,value in zip(options,score)})
                print(axis,start+len(subset),len(rows),flush=True)
    for row in rows:
        votes = row['votes']
        row['joint'] = JOINT[votes['joint']['choice']]
        row['factored'] = FACTORED.get((votes['scope']['choice'],votes['direction']['choice']),'defer')
    metrics = {}
    for method in ('joint','factored'):
        summary = {}
        for source,subset in [('all',rows)]+[(s,[r for r in rows if r['source']==s]) for s in sorted({r['source'] for r in rows})]:
            semantics = {r['id'] for r in subset}
            summary[source] = dict(rows=len(subset),correct_rows=sum(r[method]==r['expected'] for r in subset),
                exact_bilingual_semantics=sum(all(r[method]==r['expected'] for r in subset if r['id']==cid) for cid in semantics),
                bilingual_semantics=len(semantics),
                routed_wrong_non_defer=sum(r[method]!=r['expected'] and r[method]!='defer' for r in subset),
                expected_defer_but_routed=sum(r['expected']=='defer' and r[method]!='defer' for r in subset),
                route_needed_but_deferred=sum(r['expected']!='defer' and r[method]=='defer' for r in subset),
                predicted_routes=dict(Counter(r[method] for r in subset)))
        metrics[method] = summary
    write(EXP/'route-result.json',dict(at=stamp(),protocol_sha256=sha(path),device='cpu',dtype='float32',
        base_config_sha256=sha(BASE/'config.json'),base_index_sha256=sha(BASE/'model.safetensors.index.json'),
        metrics=metrics,rows=rows,GPU_hours=0,deployed=False))


if __name__=='__main__':
    main()

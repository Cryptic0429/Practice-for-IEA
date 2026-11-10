"""Reload the trained LoRA in the full frozen base and replay held raw requests."""
import torch
from PIL import Image
import numpy as np
from torch.nn.functional import linear
from transformers import AutoTokenizer,Qwen2_5_VLForConditionalGeneration

from functional_v1.common import ROOT,read,sha,stamp,write
from functional_v15.editor import load_rgb
from functional_v15.outcomes import execute
from functional_v27.qwen_text import BASE
from .cache import EXP,PLAN
from .train import RUN,LoRALinear

PROBES = ('p021-global_down','p165-shadow_up','p102-shadow_down',
          'p457-global_up','held-text-10')


def main():
    p = read(PLAN)
    result_path = EXP/'result.json'
    result = read(result_path)
    receipt = read(RUN/'receipt.json')
    adapter_path = RUN/'adapter.pt'
    if sha(adapter_path)!=receipt['adapter_sha256'] or sha(PLAN)!=receipt['protocol_sha256']:
        raise ValueError('Unexpected adapter')
    adapter = torch.load(adapter_path,map_location='cpu',weights_only=True)
    heads_path = RUN/'heads.pt'
    if sha(heads_path)!=receipt['heads_sha256']:
        raise ValueError('Unexpected supervised head')
    head = torch.load(heads_path,map_location='cpu',weights_only=True)['augmented']
    rows = [r for r in result['rows'] if r['id'] in PROBES]
    if len(rows)!=2*len(PROBES):
        raise ValueError('Smoke probes missing')
    torch.set_num_threads(p['cpu_threads'])
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    tokenizer.padding_side = 'right'
    ids = [tokenizer.encode(x,add_special_tokens=False)[0] for x in p['labels']]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE,torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).eval()
    attn = model.model.layers[-1].self_attn
    original = {key:getattr(attn,key) for key in ('q_proj','v_proj')}
    adapted = {}
    for key,projection in original.items():
        layer = LoRALinear(projection,adapter['rank'],adapter['alpha'])
        with torch.no_grad():
            for name in ('a','b'):
                getattr(layer,name).copy_(adapter['parameters'][f'layer.self_attn.{key}.{name}'])
        adapted[key] = layer
    if any(param.device.type!='cpu' for param in model.parameters()):
        raise ValueError('CPU-only replay violated')
    boards = {c['id']:c for c in read(EXP/'replay.json')['cases']}
    result_rows = []
    choices = list(p['labels'].values())
    destination = EXP/'runtime-outputs'
    destination.mkdir(exist_ok=True)
    with torch.inference_mode():
        for start in range(0,len(rows),5):
            batch = rows[start:start+5]
            texts = [tokenizer.apply_chat_template([dict(role='system',content=p['system_prompt']),
                dict(role='user',content=r['instruction'])],tokenize=False,add_generation_prompt=True) for r in batch]
            tokens = tokenizer(texts,padding=True,add_special_tokens=False,return_tensors='pt')
            inp,mask = tokens.input_ids,tokens.attention_mask
            pos,_ = model.get_rope_index(inp,None,None,mask)
            for method,projections in [('augmented_head',original),('tail_lora',adapted)]:
                for key,layer in projections.items():
                    setattr(attn,key,layer)
                hidden = model.model(input_ids=inp,attention_mask=mask,position_ids=pos,use_cache=False).last_hidden_state
                final = hidden[torch.arange(len(batch)),mask.sum(dim=1)-1]
                if method=='tail_lora':
                    scores = linear(final,model.lm_head.weight[ids])
                else:
                    normalized = ((final-head['mean'])/head['scale']).to(head['weight'].dtype)
                    scores = linear(normalized,head['weight'],head['bias'])
                for r,score in zip(batch,scores):
                    route = choices[int(score.argmax())]
                    reference = r['predictions'][method]
                    diff = max(abs(reference['logits'][key]-float(value)) for key,value in zip(choices,score))
                    if route!=reference['route'] or diff>1e-3:
                        raise ValueError(f"Full-model runtime mismatch: {r['id']}/{r['language']}")
                    row = dict(id=r['id'],language=r['language'],method=method,route=route,
                               logit_max_error=diff,expected=r['expected'],output=None)
                    if route!='defer' and r['cohort']=='held_photos':
                        case = boards[r['group']]
                        src = ROOT/case['source']
                        if sha(src)!=case['source_sha256']:
                            raise ValueError('Original photo changed')
                        pixels = execute(load_rgb(src),dict(action='edit',parameters=p['editor_routes'][route]))['result']
                        if not np.array_equal(pixels,load_rgb(ROOT/reference['output'])):
                            raise ValueError('Full-model edit path did not reproduce output pixels')
                        out = destination/f"{r['id']}-{r['language']}-{method}.png"
                        Image.fromarray(pixels).save(out)
                        row.update(output=str(out.relative_to(ROOT)),sha256=sha(out),pixels_equal=True,
                                   visual_reference=reference['visual'])
                    result_rows.append(row)
    write(EXP/'runtime-verification.json',dict(at=stamp(),result_sha256=sha(result_path),
        adapter_sha256=sha(adapter_path),heads_sha256=sha(heads_path),inputs=len(rows),rows=result_rows,
        max_logit_error=max(r['logit_max_error'] for r in result_rows),GPU_hours=0,deployed=False))


if __name__ == '__main__':
    main()

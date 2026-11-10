"""Cache target-free frozen Qwen text activations for CPU tail adaptation."""
from collections import Counter
from pathlib import Path
import unicodedata

import torch
from torch.nn.functional import linear
from transformers import AutoTokenizer,Qwen2_5_VLForConditionalGeneration

from functional_v1.common import ROOT,read,sha,stamp,write
from functional_v27.qwen_text import BASE

EXP = ROOT/'experiments/functional-v29-20261107'
DATA = ROOT/'data/training/functional-v29-20261107'
PLAN = EXP/'protocol.json'
TRAIN = {'legacy','augmented'}


def rows_and_plan():
    plan = read(PLAN)
    path = DATA/'requests.json'
    if sha(path)!=plan['requests_sha256']:
        raise ValueError('Frozen request file changed')
    cases = read(path)
    literal = lambda s:' '.join(unicodedata.normalize('NFKC',s).casefold().split())
    learn = {literal(s) for r in cases if r['cohort'] in TRAIN for s in r['requests'].values()}
    test = {literal(s) for r in cases if r['cohort'] not in TRAIN for s in r['requests'].values()}
    overlap = learn & test
    if overlap:
        raise ValueError(f'Literal requests overlap fit and test: {sorted(overlap)}')
    rows = [dict(id=r['id'],group=r['group'],cohort=r['cohort'],language=l,instruction=s,expected=r['expected'])
            for r in cases for l,s in r['requests'].items()]
    return rows,plan


def main():
    rows,plan = rows_and_plan()
    folder = EXP/'cache'
    folder.mkdir(exist_ok=True)
    if (folder/'manifest.json').exists():
        raise FileExistsError('Keep first cached extraction')
    torch.set_num_threads(plan['cpu_threads'])
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    tokenizer.padding_side = 'right'
    ids = [tokenizer.encode(c,add_special_tokens=False) for c in plan['labels']]
    if not all(len(v)==1 for v in ids):
        raise ValueError('Option labels must be single tokens')
    ids = [x[0] for x in ids]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE,torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).eval()
    if any(p.device.type!='cpu' for p in model.parameters()):
        raise ValueError('CPU-only extraction violated')
    choice_weights = model.lm_head.weight[ids].detach()
    tail_path = folder/'tail.pt'
    torch.save(dict(layer=model.model.layers[-1].state_dict(),norm=model.model.norm.state_dict(),
                    choices=choice_weights,tail_index=len(model.model.layers)-1),tail_path)
    batch_size = plan['cache_batch']
    blocks = []
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            subset = rows[start:start+batch_size]
            messages = [[dict(role='system',content=plan['system_prompt']),
                         dict(role='user',content=r['instruction'])] for r in subset]
            prompts = [tokenizer.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in messages]
            tokens = tokenizer(prompts,padding=True,add_special_tokens=False,return_tensors='pt')
            inp,mask = tokens.input_ids,tokens.attention_mask
            pos,_ = model.get_rope_index(inp,None,None,mask)
            result = model.model(input_ids=inp,attention_mask=mask,position_ids=pos,
                                 use_cache=False,output_hidden_states=True)
            index = mask.sum(dim=1)-1
            final = result.last_hidden_state[torch.arange(len(subset)),index]
            scores = linear(final,choice_weights)
            out = folder/f'block-{start:04d}.pt'
            torch.save(dict(prefix=result.hidden_states[-2].contiguous(),lengths=index+1,
                            final=final.contiguous(),scores=scores.contiguous()),out)
            blocks.append(dict(path=str(out.relative_to(ROOT)),sha256=sha(out),start=start,rows=len(subset)))
            print(f'Cached {start+len(subset)}/{len(rows)}',flush=True)
            write(folder/'progress.json',dict(at=stamp(),rows=start+len(subset),total=len(rows),GPU_hours=0))
    write(folder/'manifest.json',dict(at=stamp(),protocol_sha256=sha(PLAN),rows=len(rows),
        by_cohort=dict(Counter(r['cohort'] for r in rows)),blocks=blocks,
        tail_path=str(tail_path.relative_to(ROOT)),tail_sha256=sha(tail_path),
        base_config_sha256=sha(BASE/'config.json'),base_index_sha256=sha(BASE/'model.safetensors.index.json'),
        device='cpu',dtype='float32',GPU_hours=0))


if __name__ == '__main__':
    main()

"""Frozen Qwen on CPU as a text-only strength classifier."""
import torch
from torch.nn.functional import linear
from transformers import AutoTokenizer,Qwen2_5_VLForConditionalGeneration

from functional_v1.common import ROOT,read,sha,stamp,write
from functional_v26.rank import vector
from .rank import choose,summarize
from .prepare import DATA,EXP

BASE = ROOT/'models/Qwen2.5-VL-7B-Instruct'


def main():
    protocol_path = EXP/'qwen-protocol.json'
    protocol = read(protocol_path)
    result_path,fit_path = EXP/'result.json',EXP/'fit.json'
    if (sha(result_path),sha(fit_path)) != (protocol['parent_result_sha256'],protocol['parent_fit_sha256']):
        raise ValueError('Parent experiment changed')
    reference,fit = read(result_path),read(fit_path)
    cases_path = DATA/'held-cases.json'
    if sha(cases_path)!=reference['held_cases_sha256']:
        raise ValueError('Held label set changed')
    cases = read(cases_path)
    rows = [dict(id=r['id'],request=r['request'],language=r['language'],
                instruction=r['instruction'],expected='A' if r['request']=='mild' else 'B',
                kind='held_edit') for r in reference['predictions']]
    rows.extend(dict(id=c['id'],language=lang,instruction=text,expected=c['expected'],
                     kind='text_control') for c in protocol['controls']
                for lang,text in c['instructions'].items())
    torch.set_num_threads(protocol['cpu_threads'])
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    ids = [tokenizer.encode(letter,add_special_tokens=False) for letter in 'ABC']
    if not all(len(x)==1 for x in ids):
        raise ValueError('Expected single-token A/B/C options')
    ids = [x[0] for x in ids]
    print('Loading frozen Qwen on CPU',flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE,torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).eval()
    if any(p.device.type!='cpu' for p in model.parameters()):
        raise ValueError('CPU-only path violated')
    prompts = [tokenizer.apply_chat_template([
        dict(role='system',content=protocol['prompt']),dict(role='user',content=r['instruction'])
    ],tokenize=False,add_generation_prompt=True) for r in rows]
    option_weights = model.lm_head.weight[ids]
    batch = protocol['batch_size']
    with torch.inference_mode():
        for start in range(0,len(rows),batch):
            tokens = tokenizer(prompts[start:start+batch],add_special_tokens=False,
                               padding=True,return_tensors='pt')
            inp,mask = tokens.input_ids,tokens.attention_mask
            pos,_ = model.get_rope_index(inp,None,None,mask)
            hidden = model.model(input_ids=inp,attention_mask=mask,position_ids=pos,use_cache=False).last_hidden_state
            final = hidden[torch.arange(inp.shape[0]),mask.sum(dim=1)-1]
            logits = linear(final,option_weights)
            for row,score in zip(rows[start:start+batch],logits):
                row.update(logits={label:float(s) for label,s in zip('ABC',score)},
                           selected='ABC'[int(score.argmax())])
            print(f'Scored {min(len(rows),start+batch)}/{len(rows)}',flush=True)
    by_id = {c['id']:c for c in cases}
    values = {c['id']:[vector(c,v) for v in c['variants']] for c in cases}
    weights = torch.tensor(fit['weights']['conditional_pixels'],dtype=torch.float64).numpy()
    predicted = []
    for r in rows:
        if r['kind']!='held_edit':
            continue
        case = by_id[r['id']]
        intent = {'A':-1,'B':1,'C':None}[r['selected']]
        selected = {'qwen_preset':None if intent is None else int(intent>0),
                    'qwen_pixels':choose(values[r['id']],weights,True,11,intent)}
        predicted.append(dict(r,choices={name:dict(choice=None if i is None else case['variants'][i]['id'],
            verdict='abstained' if i is None else case['labels'][r['request']][i]) for name,i in selected.items()}))
    controls = [r for r in rows if r['kind']=='text_control']
    write(EXP/'qwen-result.json',dict(at=stamp(),protocol_sha256=sha(protocol_path),parent_result_sha256=sha(result_path),
        cpu_threads=protocol['cpu_threads'],device='cpu',model=str(BASE.relative_to(ROOT)),
        base_config_sha256=sha(BASE/'config.json'),base_index_sha256=sha(BASE/'model.safetensors.index.json'),
        tokenizer_sha256=sha(BASE/'tokenizer.json'),correct_text_controls=sum(r['selected']==r['expected'] for r in controls),
        correct_held_intents=sum(r['selected']==r['expected'] for r in predicted),
        summary=summarize(predicted,cases),controls=controls,predictions=predicted,GPU_hours=0,deployed=False))


if __name__ == '__main__':
    main()

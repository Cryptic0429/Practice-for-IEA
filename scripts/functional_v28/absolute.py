"""Check absolute visual judgments after detecting pair position bias."""
import torch
from PIL import Image,ImageDraw
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration

from functional_v1.common import ROOT,read,sha,stamp,write
from functional_v27.qwen_text import BASE
from .boards import EXP,PROTOCOL
from .score import last_logits


def main():
    plan_path = EXP/'absolute-protocol.json'
    plan = read(plan_path)
    if (plan['pair_result_sha256'],plan['boards_sha256']) != (sha(EXP/'result.json'),sha(EXP/'boards.json')):
        raise ValueError('Pair comparison state changed')
    images = {c['id']:c['boards'] for c in read(EXP/'boards.json')['cases']}
    reference = {c['id']:c['expected'] for c in read(EXP/'reference.json')['cases']}
    requests = {c['id']:c['requests'][plan['language']] for c in read(PROTOCOL)['cases']}
    rows = []
    folder = EXP/'absolute-boards'
    folder.mkdir(exist_ok=True)
    for cid in plan['subset']:
        for candidate,order in [('1','12'),('2','21')]:
            source = ROOT/images[cid][order]['path']
            if sha(source)!=images[cid][order]['sha256']:
                raise ValueError('Original board changed')
            with Image.open(source) as opened:
                board = opened.crop((0,0,448,364)).convert('RGB')
            draw = ImageDraw.Draw(board)
            draw.rectangle((224,0,447,26),fill='white')
            draw.text((229,7),'CANDIDATE',fill='black')
            path = folder/f'{cid}-{candidate}.png'
            board.save(path)
            rows.append(dict(id=cid,candidate=candidate,path=str(path.relative_to(ROOT)),sha256=sha(path),
                             instruction=requests[cid],expected='Yes' if reference[cid]==candidate else 'No'))
    torch.set_num_threads(plan['cpu_threads'])
    torch.set_num_interop_threads(1)
    processor = AutoProcessor.from_pretrained(BASE,use_fast=False,local_files_only=True)
    processor.tokenizer.padding_side = 'right'
    ids = [processor.tokenizer.encode(x,add_special_tokens=False) for x in ('Yes','No')]
    if not all(len(v)==1 for v in ids):
        raise ValueError('Expected one token per class')
    ids = [v[0] for v in ids]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE,torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).eval()
    if any(p.device.type!='cpu' for p in model.parameters()):
        raise ValueError('CPU-only execution violated')
    batch = plan['batch_size']
    with torch.inference_mode():
        for start in range(0,len(rows),batch):
            imgs,prompts = [],[]
            subset = rows[start:start+batch]
            for r in subset:
                with Image.open(ROOT/r['path']) as opened:
                    imgs.append(opened.convert('RGB'))
                messages = [dict(role='system',content=plan['system_prompt']),
                    dict(role='user',content=[dict(type='image'),dict(type='text',text=r['instruction'])])]
                prompts.append(processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
            inputs = processor(text=prompts,images=imgs,padding=True,return_tensors='pt')
            logits = last_logits(model,inputs)
            for r,logit in zip(subset,logits):
                score = logit[ids]
                r['logits'] = {s:float(v) for s,v in zip(('Yes','No'),score)}
                r['selected'] = ('Yes','No')[int(score.argmax())]
                r['unconstrained_top5'] = [processor.tokenizer.decode([int(v)])
                                           for v in logit.topk(5).indices]
            print(f'Scored {start+len(subset)}/{len(rows)}',flush=True)
    write(EXP/'absolute-result.json',dict(at=stamp(),protocol_sha256=sha(plan_path),device='cpu',
        correct=sum(r['expected']==r['selected'] for r in rows),
        approved_bad=sum(r['expected']=='No' and r['selected']=='Yes' for r in rows),
        refused_good=sum(r['expected']=='Yes' and r['selected']=='No' for r in rows),
        rows=rows,GPU_hours=0,deployed=False))


if __name__ == '__main__':
    main()

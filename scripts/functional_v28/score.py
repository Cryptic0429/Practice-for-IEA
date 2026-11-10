"""Frozen image comparison with the project's local Qwen2.5-VL on CPU."""
import time

import torch
from PIL import Image
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration

from functional_v1.common import ROOT,read,sha,stamp,write
from functional_v27.qwen_text import BASE
from .boards import EXP,PROTOCOL


def last_logits(model,inputs):
    last = inputs['attention_mask'].sum(dim=1)-1
    rows = torch.arange(len(last))

    def final_state_only(_layer,args):
        # Prune unused time positions for the frozen LM head.
        return (args[0][rows,last].unsqueeze(1),)

    handle = model.lm_head.register_forward_pre_hook(final_state_only)
    try:
        return model(**inputs,use_cache=False).logits[:,0,:]
    finally:
        handle.remove()


def main():
    output = EXP/'scores.json'
    if output.exists():
        raise FileExistsError('Keep the first evaluation frozen')
    protocol,manifest = read(PROTOCOL),read(EXP/'boards.json')
    if manifest['protocol_sha256']!=sha(PROTOCOL):
        raise ValueError('Visual protocol changed')
    reference_path = EXP/'reference.json'
    reference = read(reference_path)
    if reference['boards_sha256']!=sha(EXP/'boards.json') or reference['protocol_sha256']!=sha(PROTOCOL):
        raise ValueError('Frozen reference mismatch')
    reference_sha = sha(reference_path)
    del reference
    choices = {c['id']:c['requests'] for c in protocol['cases']}
    rows = []
    for c in manifest['cases']:
        for order,b in c['boards'].items():
            if sha(ROOT/b['path'])!=b['sha256']:
                raise ValueError(f"Comparison changed: {b['path']}")
            for lang,text in choices[c['id']].items():
                rows.append(dict(id=c['id'],order=order,language=lang,instruction=text,
                                 board=b['path'],board_sha256=b['sha256']))
    torch.set_num_threads(protocol['cpu_threads'])
    torch.set_num_interop_threads(1)
    processor = AutoProcessor.from_pretrained(BASE,local_files_only=True)
    processor.tokenizer.padding_side = 'right'
    option_ids = [processor.tokenizer.encode(label,add_special_tokens=False) for label in 'ABC']
    if not all(len(ids)==1 for ids in option_ids):
        raise ValueError('Expected single-token option IDs')
    option_ids = [ids[0] for ids in option_ids]
    print('Loading frozen multimodal model on CPU',flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE,torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).eval()
    if any(p.device.type!='cpu' for p in model.parameters()):
        raise ValueError('CPU-only execution violated')
    batch_size = protocol['batch_size']
    timings = []
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            start_time = time.monotonic()
            batch = rows[start:start+batch_size]
            imgs,prompts = [],[]
            for row in batch:
                with Image.open(ROOT/row['board']) as opened:
                    imgs.append(opened.convert('RGB'))
                messages = [dict(role='system',content=protocol['system_prompt']),
                            dict(role='user',content=[dict(type='image'),dict(type='text',text=row['instruction'])])]
                prompts.append(processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
            inputs = processor(text=prompts,images=imgs,padding=True,return_tensors='pt')
            logits = last_logits(model,inputs)[:,option_ids]
            for row,score,grid in zip(batch,logits,inputs['image_grid_thw']):
                row.update(logits={label:float(s) for label,s in zip('ABC',score)},
                           selected='ABC'[int(score.argmax())],image_grid_thw=grid.tolist())
            timings.append(dict(start=start,rows=len(batch),seconds=time.monotonic()-start_time))
            print(f'Scored {start+len(batch)}/{len(rows)} in {timings[-1]["seconds"]:.1f}s',flush=True)
            write(EXP/'score-progress.json',dict(at=stamp(),completed=start+len(batch),total=len(rows),GPU_hours=0))
    if sha(reference_path)!=reference_sha:
        raise ValueError('Frozen reference changed during execution')
    write(output,dict(at=stamp(),protocol_sha256=sha(PROTOCOL),boards_sha256=sha(EXP/'boards.json'),
        reference_sha256=reference_sha,base_config_sha256=sha(BASE/'config.json'),
        base_index_sha256=sha(BASE/'model.safetensors.index.json'),base_model_info_sha256=sha(BASE/'model-info.json'),
        model=str(BASE.relative_to(ROOT)),cpu_threads=protocol['cpu_threads'],
        device='cpu',dtype='float32',timings=timings,rows=rows,GPU_hours=0))


if __name__ == '__main__':
    main()

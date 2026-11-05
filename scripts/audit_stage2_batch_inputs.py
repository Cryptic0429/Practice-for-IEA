"""Verify actual TRL-style padded16-row multimodal packing against unbatched inputs."""
import copy
import json
import torch
from trl.data_utils import prepare_multimodal_messages,maybe_apply_chat_template
from stage2_runtime import ROOT,processor,encoded,image_input


def main():
    rows=json.loads((ROOT/'data/stage2/stage2-A-edit-grpo-v1/preflight.json').read_text())
    p=processor();torch.set_num_threads(4);checks=[]
    for start in range(0,len(rows),4):
        group=[row for row in rows[start:start+4] for _ in range(4)]
        prompts=[]
        for row in group:
            messages=[dict(role=m['role'],content=m['content'].replace('<image>',''))
                      for m in row['conversations'] if m['role'] in ('system','user')]
            prepare_multimodal_messages(messages,num_images=1)
            prompts.append(maybe_apply_chat_template(dict(prompt=messages),p)['prompt'])
        batch=p(text=prompts,images=[[image_input(r['images'][0])] for r in group],return_tensors='pt',
                padding=True,padding_side='left',add_special_tokens=False)
        offset=0
        for i,row in enumerate(group):
            single=encoded(p,row);count=int(batch['image_grid_thw'][i].prod())
            ids=batch['input_ids'][i][batch['attention_mask'][i].bool()]
            check=dict(id=row['id'],tokens_equal=torch.equal(ids,single['input_ids'][0]),
                       grid_equal=torch.equal(batch['image_grid_thw'][i:i+1],single['image_grid_thw']),
                       pixels_equal=torch.equal(batch['pixel_values'][offset:offset+count],single['pixel_values']))
            checks.append(check);offset+=count
    result=dict(passed=all(c['tokens_equal'] and c['grid_equal'] and c['pixels_equal'] for c in checks),
                checks=checks,model_forwards=0,candidates=128,
                scope='Reconstructed TRL-style CPU packing; no capture of live GPU batches')
    out=ROOT/'logs/stage2-A-edit-grpo-v1/batch-input-parity.json'
    assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
    print({'passed':result['passed'],'candidates':128});assert result['passed']


if __name__=='__main__':
    main()

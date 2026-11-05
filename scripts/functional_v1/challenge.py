"""Recorded usability probes, separate from frozen development and acceptance sets."""
import argparse
from .common import ROOT, EXP, GPUJob, write, append, sha, stamp
from .catalog import resolve

PROBES = [
    ('brown-zh','请添加棕褐色复古色调，保留明暗层次。'),
    ('brown-en','Please add a brown vintage tone without making the image brighter.'),
    ('identity-zh','这张照片保持原样，不要做任何调整。'),
    ('identity-en','Keep this picture exactly as it is. Do not enhance anything.'),
    ('cool-zh','整体稍微调冷一点，不要改变亮度和对比度。'),
    ('warm-zh','整体稍微调暖一点，不要改变亮度和对比度。'),
    ('muted-en','Make the colours much less intense, but keep a little colour.'),
    ('richer-en','Make the colours slightly more vibrant, keeping them natural.'),
    ('richer-strong-en','Make the colours much more vibrant than they are now.'),
    ('shadow-zh','树下太黑了，把暗部提亮一些，同时保留天空的细节。'),
    ('compound-en','Warm the image slightly and reduce the saturation a little; keep brightness unchanged.'),
]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--gpu',required=True)
    args=ap.parse_args();spec=resolve(args.model)
    folder=EXP/'challenge'/args.model
    assert not (folder/'predictions.jsonl').exists(), 'Keep previous probe results immutable'
    image=ROOT/'data/images/7QSfy_7QSfy.jpg'
    with GPUJob(args.gpu,'challenge-'+args.model,max_seconds=1800) as budget:
        import torch
        from .runtime import load_model,processor,predict
        from .editor import load_rgb,edit,save
        torch.set_num_threads(4)
        model,p=load_model(spec['adapter']),processor()
        write(folder/'identity.json',{'model_id':spec['id'],'adapter':spec['adapter'],
            'adapter_sha256':sha(__import__('pathlib').Path(spec['adapter'])/'adapter_model.safetensors') if spec['adapter'] else None,
            'image':str(image.relative_to(ROOT)),'input_sha256':sha(image),
            'purpose':'Previously viewed case-12 regression and natural wording probes; not independent acceptance.',
            'no_training_use':True})
        for pid,instruction in PROBES:
            budget.tick();case={'image':str(image.relative_to(ROOT)),'instructions':{'original':instruction}}
            row={'id':pid,'instruction':instruction,'at':stamp()}
            try:
                response,params=predict(model,p,case,'original')
                output=folder/(pid+'.png');save(output,edit(load_rgb(image),params))
                row.update(status='executed',parameters=params,response=response,image=str(output.relative_to(ROOT)),sha256=sha(output))
            except Exception as e:row.update(status='failed',error=str(e))
            append(folder/'predictions.jsonl',row)


if __name__=='__main__':main()

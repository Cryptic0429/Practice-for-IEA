"""CPU comparison of all 32 preflight inputs across Stage 1 and Stage 2 stacks."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT=Path('/mnt/sdc/zhc/IEA')
OUT=ROOT/'logs/stage2-A-edit-grpo-v1'


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--export-stage1', action='store_true')
    args=cli.parse_args()
    rows=json.loads((ROOT/'data/stage2/stage2-A-edit-grpo-v1/preflight.json').read_text())
    torch.set_num_threads(4)
    if args.export_stage1:
        from llamafactory.hparams import get_infer_args
        from llamafactory.model import load_tokenizer
        from llamafactory.data import get_template_and_fix_tokenizer
        from llamafactory.chat.hf_engine import HuggingfaceEngine
        protocol=json.loads((ROOT/'configs/evaluation-protocol-v1.json').read_text())
        config={**protocol['model_common'],**protocol['models']['A'],**protocol['generation']}
        config.pop('num_return_sequences')
        ma,da,fa,ga=get_infer_args(config)
        module=load_tokenizer(ma)
        tokenizer,processor=module['tokenizer'],module['processor']
        template=get_template_and_fix_tokenizer(tokenizer,da)
        fake=SimpleNamespace(device=torch.device('cpu'),dtype=torch.bfloat16,
                             config=SimpleNamespace(model_type='qwen2_5_vl'))
        probes=[]
        for r in rows:
            inputs,_=HuggingfaceEngine._process_args(fake,tokenizer,processor,template,ga.to_dict(),
                [m for m in r['conversations'] if m['role']=='user'],
                system=next(m['content'] for m in r['conversations'] if m['role']=='system'),
                images=r['images'],input_kwargs=dict(protocol['generation']))
            probes.append(dict(id=r['id'],tensors={k:v.cpu() for k,v in inputs.items() if torch.is_tensor(v)}))
        dest=OUT/'stage1-input-probes.pt'
        assert not dest.exists()
        torch.save(probes,dest)
        print('Exported',len(probes))
    else:
        from stage2_runtime import processor,encoded
        p=processor()
        old=torch.load(OUT/'stage1-input-probes.pt',weights_only=True)
        checks=[]
        for row,probe in zip(rows,old):
            assert row['id']==probe['id']
            new=encoded(p,row)
            tensors={}
            for k,v in probe['tensors'].items():
                new_key='input_ids' if k=='inputs' else k
                match=new_key in new and torch.equal(v,new[new_key].to(v.dtype))
                tensors[k]=dict(equal=match,old_shape=list(v.shape),
                               new_shape=list(new[new_key].shape) if new_key in new else None)
            checks.append(dict(id=row['id'],tensors=tensors))
        result=dict(all_equal=all(t['equal'] for c in checks for t in c['tensors'].values()),
                    probes=checks,scope='32 training preflight rows; original Stage 1 CPU tensors including BF16 image casting',
                    model_forwards=0)
        (OUT/'input-parity.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
        assert result['all_equal']


if __name__=='__main__':
    main()

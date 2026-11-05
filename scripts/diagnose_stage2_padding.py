"""Frozen-A training probes: compare padded/unpadded generation; zero updates."""
from datetime import datetime, timezone
import argparse
import json
import os
import subprocess
import time
import traceback
import torch
from trl.models.utils import peft_module_casting_to_bf16
from stage2_runtime import ROOT, load_model, processor, encoded, adapter_state, tensor_digest, chunk_generation


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--fixed',action='store_true')
    cli.add_argument('--all-rows',action='store_true')
    cli.add_argument('--concurrency',type=int,choices=[1,2,4],default=1)
    cli.add_argument('--tokens',type=int,default=48)
    args=cli.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    assert int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used',
                                      '--format=csv,noheader,nounits'], text=True)) < 512
    name='padding-diagnostic-fixed-all' if args.all_rows else ('padding-diagnostic-fixed' if args.fixed else 'padding-diagnostic')
    out = ROOT/'outputs/stage2-A-edit-grpo-v1'/name
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    status = dict(status='running', pid=os.getpid(), gpu=0, started_at=datetime.now(timezone.utc).isoformat())
    def save():
        tmp=out/'status.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(out/'status.json')
    save()
    try:
        torch.set_num_threads(4)
        rows = {r['id']:r for r in json.loads((ROOT/'data/stage2/stage2-A-edit-grpo-v1/preflight.json').read_text())}
        candidates=[json.loads(s) for s in (ROOT/'outputs/stage2-A-edit-grpo-v1/preflight-v7/candidates.jsonl').read_text().splitlines()][:16]
        ids=list(dict.fromkeys(r['id'] for r in candidates))
        if args.all_rows:
            assert args.fixed
            ids=list(rows)
        p=processor();model=load_model();peft_module_casting_to_bf16(model);model.eval()
        if args.fixed:
            chunk_generation(model,concurrency=args.concurrency)
        initial=tensor_digest(adapter_state(model))
        inputs=[encoded(p,rows[i]) for i in ids];width=max(x['input_ids'].shape[1] for x in inputs)
        checks=[]
        for sample, original in zip(ids,inputs):
            variants=[]
            for padded in [False,True]:
                batch={k:v.repeat(args.concurrency,*([1]*(v.ndim-1))).to('cuda') for k,v in original.items()}
                if padded:
                    n=width-batch['input_ids'].shape[1]
                    batch['input_ids']=torch.nn.functional.pad(batch['input_ids'],(n,0),value=p.tokenizer.pad_token_id)
                    batch['attention_mask']=torch.nn.functional.pad(batch['attention_mask'],(n,0),value=0)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    generated=model.generate(**batch,do_sample=False,max_new_tokens=args.tokens,use_cache=True,
                        disable_compile=True,pad_token_id=p.tokenizer.pad_token_id,eos_token_id=p.tokenizer.eos_token_id)
                tokens=generated[0,batch['input_ids'].shape[1]:].tolist()
                assert all(x.tolist()==tokens for x in generated[:,batch['input_ids'].shape[1]:])
                variants.append(dict(padded=padded,tokens=tokens,text=p.decode(tokens,skip_special_tokens=True)))
            checks.append(dict(id=sample,input_tokens=original['input_ids'].shape[1],padded_width=width,
                               exact_tokens_equal=variants[0]['tokens']==variants[1]['tokens'],variants=variants))
        assert tensor_digest(adapter_state(model))==initial
        report=dict(training_updates=0,policy_unchanged=True,checks=checks,
                    passed=all(c['exact_tokens_equal'] for c in checks),concurrency=args.concurrency,
                    prompts=len(ids),max_new_tokens=args.tokens,
                    scope='Training prompts only; deterministic padding sensitivity diagnostic,not test evaluation.')
        (out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False,indent=2))
        status['status']='completed'
    except BaseException as exc:
        status.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        status.update(updated_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start)
        save()


if __name__=='__main__':
    main()

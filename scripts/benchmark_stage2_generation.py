"""Training-only resource probe:4 completions each at generation concurrency1/2/4."""
from datetime import datetime,timezone
import json
import os
import subprocess
import time
import traceback
import torch
from transformers import set_seed
from trl.models.utils import peft_module_casting_to_bf16
from stage2_runtime import ROOT,load_model,processor,prompt,image_input,adapter_state,tensor_digest
from evaluation_core import parse_parameters


def write(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    assert int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used',
                                     '--format=csv,noheader,nounits'],text=True))<512
    out=ROOT/'outputs/stage2-A-edit-grpo-v1/generation-resource-probe'
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    status=dict(status='loading',pid=os.getpid(),gpu=0,started_at=datetime.now(timezone.utc).isoformat())
    write(out/'status.json',status)
    samples=[]
    monitor=subprocess.Popen(['nvidia-smi','-i','0','--query-gpu=timestamp,memory.used,temperature.gpu,utilization.gpu',
                              '--format=csv,noheader,nounits','--loop=1'],stdout=(out/'gpu.csv').open('w'))
    try:
        torch.set_num_threads(4)
        row=json.loads((ROOT/'data/stage2/stage2-A-edit-grpo-v1/preflight.json').read_text())[0]
        protocol=json.loads((ROOT/'configs/evaluation-protocol-v1.json').read_text())
        p=processor();model=load_model();peft_module_casting_to_bf16(model);model.eval()
        initial=tensor_digest(adapter_state(model))
        text=p.apply_chat_template(prompt(row),tokenize=False,add_generation_prompt=True)
        image=image_input(row['images'][0])
        def inputs(n):
            return {k:v.to('cuda') for k,v in p(text=[text]*n,images=[image]*n,return_tensors='pt',
                                              padding=True,add_special_tokens=False).items()}
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            model.generate(**inputs(1),max_new_tokens=1,do_sample=False,disable_compile=True)
        for concurrency in [1,2,4]:
            set_seed(20261101);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            status.update(status='running',concurrency=concurrency,updated_at=datetime.now(timezone.utc).isoformat())
            write(out/'status.json',status)
            batch=inputs(concurrency);torch.cuda.synchronize();begin=time.monotonic();records=[]
            for _ in range(4//concurrency):
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    generated=model.generate(**batch,do_sample=True,temperature=.8,top_p=.9,top_k=0,
                         max_new_tokens=512,repetition_penalty=1.,use_cache=True,disable_compile=True,
                         pad_token_id=p.tokenizer.pad_token_id,eos_token_id=p.tokenizer.eos_token_id)
                for ids in generated[:,batch['input_ids'].shape[1]:].tolist():
                    if p.tokenizer.eos_token_id in ids:
                        ids=ids[:ids.index(p.tokenizer.eos_token_id)+1]
                    response=p.decode(ids,skip_special_tokens=True)
                    records.append(dict(id=row['id'],completion_ids=ids,response=response,
                                        parsing=parse_parameters(response,protocol)))
            torch.cuda.synchronize();seconds=time.monotonic()-begin
            sample=dict(concurrency=concurrency,seconds=seconds,completions=4,
                  total_output_tokens=sum(len(r['completion_ids']) for r in records),
                  tokens_per_second=sum(len(r['completion_ids']) for r in records)/seconds,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,records=records)
            samples.append(sample);write(out/'measurements.json',dict(samples=samples))
            print({k:v for k,v in sample.items() if k!='records'},flush=True)
            assert tensor_digest(adapter_state(model))==initial
        write(out/'result.json',dict(passed=True,samples=samples,training_updates=0,
              source_id=row['id'],warmup_output_tokens=1,policy_unchanged=True,
              scope='One training prompt; resource feasibility,not a quality comparison or general throughput guarantee'))
        status['status']='completed'
    except BaseException as exc:
        status.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        status.update(updated_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
                      gpu_hours=(time.monotonic()-start)/3600)
        write(out/'status.json',status);monitor.terminate();monitor.wait(timeout=10)


if __name__=='__main__':
    main()

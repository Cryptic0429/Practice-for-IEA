"""A continued-SFT control: same512 rows, one pass, batch1 x accumulation4 =128 updates."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import time
import traceback
import torch
from transformers import Trainer,TrainerCallback,TrainingArguments,set_seed
from trl.models.utils import peft_module_casting_to_bf16
from stage2_runtime import ROOT,load_model,processor,prompt,image_input,encoded,adapter_state,tensor_digest,assert_trainable


def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def build_batch(p,row):
    target=next(m['content'] for m in row['conversations'] if m['role']=='assistant')
    messages=prompt(row)+[dict(role='assistant',content=target)]
    full=p(text=[p.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)],
           images=[image_input(row['images'][0])],return_tensors='pt',add_special_tokens=False)
    prefix=encoded(p,row)['input_ids']
    n=prefix.shape[1]
    assert torch.equal(full['input_ids'][:,:n],prefix)
    assert full['input_ids'].shape[1]<=2048, 'No silent target truncation'
    full['labels']=full['input_ids'].clone()
    full['labels'][:,:n]=-100
    return full


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--max-gpu-seconds',type=float,required=True)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    assert int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used',
                                      '--format=csv,noheader,nounits'],text=True))<512
    gate=json.loads((ROOT/'configs/stage2-formal-gate.json').read_text())
    assert gate['passed']
    for path,digest in gate['evidence'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    from stage2_budget import ledger
    resources=ledger()
    assert resources['budget_safe_to_launch']
    assert 0<args.max_gpu_seconds<resources['remaining_gpu_hours']*3600
    data=ROOT/'data/stage2/stage2-A-edit-grpo-v1'
    manifest=json.loads((data/'manifest.json').read_text())
    source=data/'train.json'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['files']['train']['sha256']
    rows=json.loads(source.read_text());assert len(rows)==512
    out=ROOT/'outputs/stage2-A-edit-grpo-v1/sft-control'
    out.mkdir(parents=True,exist_ok=False)
    (out/'source').mkdir()
    for name in ['train_stage2_sft_control.py','stage2_runtime.py']:
        shutil.copy2(ROOT/'scripts'/name,out/'source'/name)
    start=time.monotonic()
    status=dict(status='loading',pid=os.getpid(),gpu=0,started_at=datetime.now(timezone.utc).isoformat(),step=0)
    write(out/'status.json',status)
    monitor=subprocess.Popen(['nvidia-smi','-i','0','--query-gpu=timestamp,memory.used,temperature.gpu,utilization.gpu',
                               '--format=csv,noheader,nounits','--loop=2'],stdout=(out/'gpu.csv').open('w'))
    try:
        set_seed(20261101);torch.set_num_threads(4)
        p=processor()
        class Rows(torch.utils.data.Dataset):
            def __len__(self): return len(rows)
            def __getitem__(self,index): return rows[index]
        def collate(batch):
            assert len(batch)==1
            result=build_batch(p,batch[0])
            result['sample_ids']=[batch[0]['id']]
            return result
        # Validate all label masks before GPU loading; data remain identical to GRPO pool.
        tokens=[]
        for row in rows:
            batch=collate([row]);tokens.append(dict(id=row['id'],input_tokens=int(batch['input_ids'].numel()),
                                                   supervised_tokens=int((batch['labels']!=-100).sum())))
        write(out/'token-audit.json',dict(rows=tokens,total_supervised_tokens=sum(r['supervised_tokens'] for r in tokens),
                                         repeats_per_row=1,effective_batch=4,updates=128))
        model=load_model()
        peft_module_casting_to_bf16(model)
        initial=tensor_digest(adapter_state(model));reference=tensor_digest(adapter_state(model,'reference'))
        config=TrainingArguments(output_dir=str(out),max_steps=128,per_device_train_batch_size=1,
            gradient_accumulation_steps=4,learning_rate=1e-6,warmup_ratio=.05,lr_scheduler_type='cosine',
            optim='adamw_torch',weight_decay=0.,bf16=True,gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False},max_grad_norm=1.,
            save_strategy='steps',save_steps=64,logging_steps=1,report_to='none',remove_unused_columns=False,
            seed=20261101,data_seed=20261101,dataloader_num_workers=0,disable_tqdm=True)
        write(out/'run-config.json',config.to_dict())
        class Measured(Trainer):
            seen_samples=0
            supervised_tokens=0
            seen_ids=[]
            def training_step(self,model,inputs,num_items_in_batch=None):
                self.seen_ids.extend(inputs.pop('sample_ids'))
                self.seen_samples+=inputs['input_ids'].shape[0]
                self.supervised_tokens+=int((inputs['labels']!=-100).sum())
                result=super().training_step(model,inputs,num_items_in_batch)
                assert torch.isfinite(result).all()
                return result
        budget=args.max_gpu_seconds
        class Audit(TrainerCallback):
            def on_pre_optimizer_step(self,args,state,control,**kwargs):
                assert_trainable(model)
                grads=[p.grad for p in model.parameters() if p.requires_grad]
                assert all(g is not None and torch.isfinite(g).all() for g in grads)
                assert tensor_digest(adapter_state(model,'reference'))==reference
            def on_step_end(self,args,state,control,**kwargs):
                status.update(status='running',step=state.global_step,updated_at=datetime.now(timezone.utc).isoformat(),
                              elapsed_seconds=time.monotonic()-start)
                write(out/'status.json',status)
                if time.monotonic()-start>=budget:
                    control.should_save=True;control.should_training_stop=True
            def on_log(self,args,state,control,logs=None,**kwargs):
                with (out/'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(dict(step=state.global_step,**(logs or {})),allow_nan=False)+'\n')
        trainer=Measured(model=model,args=config,train_dataset=Rows(),data_collator=collate,callbacks=[Audit()])
        result=trainer.train()
        trainer.save_model(str(out/'final'));trainer.save_state()
        assert tensor_digest(adapter_state(model))!=initial
        model.eval()
        probe=encoded(p,rows[0])
        torch.save(dict(input=dict(probe),sample_id=rows[0]['id']),out/'reload-probe.pt')
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(**{k:v.to('cuda') for k,v in probe.items()},use_cache=False).logits[:,-1].float().cpu()
        assert torch.isfinite(logits).all()
        torch.save(logits,out/'reload-expected-logits.pt')
        write(out/'training-verification.json',dict(step=trainer.state.global_step,seen_samples=trainer.seen_samples,
             supervised_tokens=trainer.supervised_tokens,expected_supervised_tokens=sum(r['supervised_tokens'] for r in tokens),
             reference_unchanged=tensor_digest(adapter_state(model,'reference'))==reference,
             final_adapter_digest=tensor_digest(adapter_state(model)),metrics=result.metrics))
        write(out/'sample-exposures.json',dict(counts=dict(Counter(trainer.seen_ids))))
        assert tensor_digest(adapter_state(model,'reference'))==reference
        status['status']='completed_training' if trainer.state.global_step==128 else 'budget_stopped'
        if trainer.state.global_step==128:
            assert trainer.seen_samples==512 and trainer.supervised_tokens==sum(r['supervised_tokens'] for r in tokens)
            assert Counter(trainer.seen_ids)==Counter({r['id']:1 for r in rows})
    except BaseException as exc:
        status.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        status.update(updated_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
                      gpu_hours=(time.monotonic()-start)/3600)
        write(out/'status.json',status);monitor.terminate();monitor.wait(timeout=10)


if __name__=='__main__':
    main()

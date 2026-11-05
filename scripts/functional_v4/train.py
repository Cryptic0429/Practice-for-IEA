"""Matched continued SFT: only measured input descriptors differ between the two routes."""
import argparse,os,shutil,traceback
from pathlib import Path
from .common import *


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--route',choices=['plain','measured'],required=True);ap.add_argument('--gpu',choices=['0','2','3'],required=True);ap.add_argument('--seed',type=int,default=20261108);args=ap.parse_args()
    protocol=read(EXP/'protocol.json');admitted=read(DATA/'admission.json')
    assert protocol['status']=='targets_frozen_before_training'
    assert read(EXP/'input-audit.json')['status']=='passed'
    for file,digest in protocol['code_hashes'].items():assert sha(ROOT/file)==digest
    assert sha(DATA/'train-accepted.json')==protocol['dataset_sha256']==admitted['dataset_sha256']
    assert sha(DATA/'cases.json')==protocol['cases_sha256']==admitted['cases_sha256']
    assert sha(DATA/'admission.json')==protocol['admission_sha256']
    assert sha(PARENT/'adapter_model.safetensors')==protocol['parent_sha256']
    cases={c['id']:c for c in read(DATA/'cases.json')};rows=read(DATA/'train-accepted.json')
    assert len(rows)==920 and len({r['id'] for r in rows})==920
    assert all(r['parameters']==cases[r['case_id']]['parameters'] and cases[r['case_id']]['split']=='train' for r in rows)
    out=EXP/'runs'/f'{args.route}-seed-{args.seed}';out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(Path(__file__).parent,out/'source',ignore=shutil.ignore_patterns('__pycache__'))
    write(out/'protocol-snapshot.json',protocol);write(out/'settings.json',{'route':args.route,'seed':args.seed,**protocol['training'],'dataset_sha256':protocol['dataset_sha256'],'parent_sha256':protocol['parent_sha256'],'runtime_sha256':sha(ROOT/'scripts/functional_v4/runtime.py'),'input_route':args.route})
    status={'status':'starting','pid':os.getpid(),'gpu':args.gpu,'started_at':stamp()};write(out/'status.json',status)
    with GPUJob(args.gpu,'v4-train-'+args.route,max_seconds=5400) as budget:
        import torch
        from transformers import Trainer,TrainingArguments,TrainerCallback,set_seed
        from .runtime import processor,load_model,build_batch
        torch.set_num_threads(4);set_seed(args.seed);p=processor();prepared=[];audit=[]
        for r in rows:
            batch=build_batch(p,cases[r['case_id']],r['language'],r['parameters'],args.route)
            audit.append({'id':r['id'],'input_tokens':int(batch['input_ids'].numel()),'assistant_tokens':int(batch['labels'].ne(-100).sum())});prepared.append(batch)
        write(out/'token-audit.json',audit);budget.tick();model=load_model(PARENT,train=True)
        initial={n:w.detach().cpu().clone() for n,w in model.named_parameters() if w.requires_grad}
        configuration=TrainingArguments(output_dir=str(out),num_train_epochs=2,per_device_train_batch_size=1,
            gradient_accumulation_steps=4,learning_rate=5e-5,warmup_ratio=.05,lr_scheduler_type='cosine',
            optim='adamw_torch',weight_decay=0.,bf16=True,gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False},max_grad_norm=1.,save_strategy='epoch',
            logging_steps=1,report_to='none',remove_unused_columns=False,seed=args.seed,data_seed=args.seed,
            dataloader_num_workers=0,disable_tqdm=True)
        write(out/'training-arguments.json',configuration.to_dict())
        class Audit(TrainerCallback):
            def on_pre_optimizer_step(self,args,state,control,**kwargs):
                active=[w for w in model.parameters() if w.requires_grad]
                assert len(active)==112 and all(w.dtype==torch.float32 for w in active)
                assert all(w.grad is not None and torch.isfinite(w.grad).all() for w in active)
            def on_step_end(self,args,state,control,**kwargs):
                status.update(status='running',step=state.global_step,epoch=state.epoch,at=stamp());write(out/'status.json',status)
                try:budget.tick()
                except Exception:
                    control.should_training_stop=True;control.should_save=True;status['budget_stop_requested']=True
            def on_log(self,args,state,control,logs=None,**kwargs):append(out/'metrics.jsonl',{'step':state.global_step,**(logs or {})})
        try:
            trainer=Trainer(model=model,args=configuration,train_dataset=prepared,data_collator=lambda b:b[0],callbacks=[Audit()])
            result=trainer.train();trainer.save_model(str(out/'final'));trainer.save_state()
            changed=any(not torch.equal(initial[n],w.detach().cpu()) for n,w in model.named_parameters() if n in initial);assert changed
            assert status.get('budget_stop_requested') or trainer.state.global_step==460
            status.update(status='budget_stopped' if status.get('budget_stop_requested') else 'completed',step=trainer.state.global_step,
               metrics=result.metrics,weights_changed=changed,adapter_sha256=sha(out/'final/adapter_model.safetensors'),finished_at=stamp())
            write(out/'status.json',status)
        except BaseException as e:
            status.update(status='failed',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc());write(out/'status.json',status);raise
if __name__=='__main__':main()

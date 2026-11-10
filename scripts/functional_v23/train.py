"""Two fixed partial-epoch pilots; every actual language presentation is logged."""
import argparse,os,signal,shutil,hashlib,traceback
from collections import Counter
from .common import *

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--gpu',choices=['0','2','3'],required=True);a=ap.parse_args();p=protocol();s=p['training'];assert a.seed in s['seeds']
 import torch
 from transformers import Trainer,TrainingArguments,TrainerCallback,set_seed
 from .runtime import load_model
 torch.set_num_threads(4);assert sha(ROOT/p['cache_path'])==p['cache_sha256']
 cache=torch.load(ROOT/p['cache_path'],map_location='cpu',weights_only=True);rows=read(DATA/'train-accepted.json');cases={c['id']:c for c in read(DATA/'cases.json')};assert len(cache)==len(rows)==1958
 prepared=[dict(b,_row_index=i) for i,b in enumerate(cache)]
 out=EXP/'runs'/f'seed-{a.seed}';out.mkdir(parents=True,exist_ok=False);shutil.copytree(__import__('pathlib').Path(__file__).parent,out/'source',ignore=shutil.ignore_patterns('__pycache__'))
 write(out/'protocol-snapshot.json',p);state=dict(status='starting',seed=a.seed,gpu=a.gpu,pid=os.getpid(),started_at=stamp());write(out/'status.json',state)
 stop={'requested':False};signal.signal(signal.SIGTERM,lambda *_:stop.update(requested=True));signal.signal(signal.SIGINT,lambda *_:stop.update(requested=True));seen=[]
 with GPUJob(a.gpu,'v23-direct-quality-QLoRA-seed-'+str(a.seed),max_seconds=s['max_seconds_per_seed']) as budget:
  try:
   set_seed(a.seed);model=load_model(train=True,seed=a.seed);write(out/'initialization.json',model.functional_initialization)
   initial={n:v.detach().cpu().clone() for n,v in model.named_parameters() if v.requires_grad};assert len(initial)==116 and sum('visual' in n for n in initial)==4
   write(out/'initial-fingerprints.json',{n:hashlib.sha256(v.numpy().tobytes()).hexdigest() for n,v in initial.items()})
   class ActualTrainer(Trainer):
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
     index=int(inputs.pop('_row_index'));seen.append(index);r=rows[index]
     append(out/'presentations.jsonl',dict(microstep=len(seen),step=self.state.global_step,row_index=index,id=r['id'],case_id=r['case_id'],language=r['language']))
     result=model(**inputs);assert torch.isfinite(result.loss)
     return (result.loss,result) if return_outputs else result.loss
   config=TrainingArguments(output_dir=str(out),max_steps=s['max_steps'],per_device_train_batch_size=1,gradient_accumulation_steps=4,
    learning_rate=s['learning_rate'],warmup_ratio=.05,lr_scheduler_type='cosine',optim='adamw_torch',weight_decay=0.,bf16=True,
    gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},max_grad_norm=1.,save_strategy='steps',save_steps=250,
    logging_steps=1,report_to='none',remove_unused_columns=False,seed=a.seed,data_seed=a.seed,dataloader_num_workers=0,disable_tqdm=True)
   write(out/'training-arguments.json',config.to_dict())
   class Audit(TrainerCallback):
    def on_pre_optimizer_step(self,args,ts,control,**kw):
     active={n:v for n,v in model.named_parameters() if v.requires_grad};assert active.keys()==initial.keys()
     assert all(v.dtype==torch.float32 and v.grad is not None and torch.isfinite(v.grad).all() for v in active.values())
     if ts.global_step==0:
      assert {id(v) for g in kw['optimizer'].param_groups for v in g['params']}=={id(v) for v in active.values()}
      write(out/'first-gradient-check.json',dict(at=stamp(),finite=True,tensors=116,only_adapter_optimizer=True,parameters=sum(v.numel() for v in active.values())))
    def on_step_end(self,args,ts,control,**kw):
     state.update(status='running',step=ts.global_step,epoch=ts.epoch,at=stamp());write(out/'status.json',state)
     try:budget.tick()
     except BudgetExceeded:stop['requested']=True
     if stop['requested']:control.should_training_stop=True;control.should_save=True;state['budget_stop_requested']=True
    def on_log(self,args,ts,control,logs=None,**kw):append(out/'metrics.jsonl',dict(step=ts.global_step,**(logs or {})))
   trainer=ActualTrainer(model=model,args=config,train_dataset=prepared,data_collator=lambda b:b[0],callbacks=[Audit()]);trainer.model_accepts_loss_kwargs=False
   set_seed(a.seed);result=trainer.train();trainer.save_model(str(out/'final'));trainer.save_state()
   if state.get('budget_stop_requested'):trainer._save_checkpoint(model,trial=None)
   changed={n:not torch.equal(v,dict(model.named_parameters())[n].detach().cpu()) for n,v in initial.items()};assert all(torch.isfinite(v).all() for v in model.parameters() if v.requires_grad)
   selected=[rows[i] for i in seen];ids={r['case_id'] for r in selected};new={c['id'] for c in read(DATA/'new-cases.json')}
   write(out/'actual-data-coverage.json',dict(at=stamp(),eligible_language_rows=1958,actual_language_presentations=len(seen),unique_language_rows=len(set(seen)),unique_cases=len(ids),original_groups=len({cases[c]['group_id'] for c in ids}),new_cases_seen=len(ids&new),new_language_presentations=sum(r['case_id'] in new for r in selected),actions=dict(Counter(r['decision']['action'] for r in selected)),row_sequence_sha256=sha(out/'presentations.jsonl'),partial_epoch=True))
   state.update(status='budget_stopped' if state.get('budget_stop_requested') else 'completed',step=trainer.state.global_step,finished_at=stamp(),metrics=result.metrics,changed_tensors=sum(changed.values()),adapter_sha256=sha(out/'final/adapter_model.safetensors'),language_presentations=len(seen))
   if state['status']=='completed':assert trainer.state.global_step==250 and len(seen)==1000 and sum(changed.values())==116
   write(out/'status.json',state)
  except BaseException as e:
   state.update(status='failed',error=str(e),traceback=traceback.format_exc(),at=stamp());write(out/'status.json',state);raise
if __name__=='__main__':main()

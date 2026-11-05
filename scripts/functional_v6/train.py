"""Matched fresh-base and continued-parent QLoRA with an immutable admitted dataset."""
import argparse,os,shutil,traceback
from functional_v1.common import ROOT,read,write,append,sha,stamp,GPUJob
from .source_bank import EXP,DATA


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--route',choices=['base','continued'],required=True);ap.add_argument('--gpu',choices=['0','2','3'],required=True);ap.add_argument('--seed',type=int,default=20261110);a=ap.parse_args()
    protocol=read(EXP/'training-protocol.json');settings=protocol['training']
    assert protocol['status']=='actual_data_targets_runtime_optimizer_frozen_before_training'
    assert a.seed in (settings['seed'],settings['replicate_seed'])
    if a.seed==settings['replicate_seed']:
        decision=read(EXP/'development-selection.json');assert decision['qualified'] and decision['route']==a.route
    for f,digest in protocol['code_hashes'].items():assert sha(ROOT/f)==digest
    for file,key in [('train-accepted.json','dataset_sha256'),('cases.json','cases_sha256'),('admission.json','admission_sha256')]:assert sha(DATA/file)==protocol[key]
    assert sha(EXP/'input-audit.json')==protocol['input_audit_sha256']
    assert sha(EXP/'target-replay.json')==protocol['target_replay_sha256']
    assert sha(EXP/'evaluation-protocol.json')==protocol['evaluation_protocol_sha256']
    parent=ROOT/protocol['parent'];assert sha(parent/'adapter_model.safetensors')==protocol['parent_sha256']
    rows=read(DATA/'train-accepted.json');cases={c['id']:c for c in read(DATA/'cases.json')}
    assert len(rows)==settings['rows']==920
    assert all(r['parameters']==cases[r['case_id']]['parameters'] and cases[r['case_id']]['split']=='train' for r in rows)
    out=EXP/'runs'/f'{a.route}-seed-{a.seed}';out.mkdir(parents=True,exist_ok=False)
    from pathlib import Path
    shutil.copytree(Path(__file__).parent,out/'source',ignore=shutil.ignore_patterns('__pycache__'))
    write(out/'protocol-snapshot.json',protocol);write(out/'settings.json',dict(route=a.route,seed=a.seed,training=settings,
        dataset_sha256=protocol['dataset_sha256'],initial_adapter_sha256=protocol['parent_sha256'] if a.route=='continued' else None))
    status=dict(status='starting',pid=os.getpid(),gpu=a.gpu,started_at=stamp());write(out/'status.json',status)
    with GPUJob(a.gpu,'v6-train-'+a.route,max_seconds=5400) as budget:
        import torch
        from transformers import Trainer,TrainingArguments,TrainerCallback,set_seed
        from functional_v1.runtime import processor,load_model,build_batch
        torch.set_num_threads(4);set_seed(a.seed);p=processor();prepared=[];audit=[]
        for r in rows:
            c=cases[r['case_id']];assert sha(ROOT/c['image'])==c['input_sha256']
            batch=build_batch(p,c,r['language'],r['parameters'])
            audit.append(dict(id=r['id'],input_tokens=int(batch['input_ids'].numel()),assistant_tokens=int(batch['labels'].ne(-100).sum())))
            prepared.append(batch)
        write(out/'token-audit.json',audit);budget.tick()
        model=load_model(parent if a.route=='continued' else None,train=True)
        initial={n:w.detach().cpu().clone() for n,w in model.named_parameters() if w.requires_grad}
        config=TrainingArguments(output_dir=str(out),num_train_epochs=settings['epochs'],per_device_train_batch_size=1,
            gradient_accumulation_steps=4,learning_rate=settings['learning_rate'],warmup_ratio=.05,lr_scheduler_type='cosine',
            optim='adamw_torch',weight_decay=0.,bf16=True,gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False},max_grad_norm=1.,save_strategy='epoch',
            logging_steps=1,report_to='none',remove_unused_columns=False,seed=a.seed,data_seed=a.seed,
            dataloader_num_workers=0,disable_tqdm=True)
        write(out/'training-arguments.json',config.to_dict())
        class Audit(TrainerCallback):
            def on_pre_optimizer_step(self,args,state,control,**kwargs):
                active=[w for w in model.parameters() if w.requires_grad]
                assert len(active)==112 and all(w.dtype==torch.float32 for w in active)
                assert all(w.grad is not None and torch.isfinite(w.grad).all() for w in active)
            def on_step_end(self,args,state,control,**kwargs):
                status.update(status='running',step=state.global_step,epoch=state.epoch,at=stamp());write(out/'status.json',status)
                try:budget.tick()
                except Exception:control.should_training_stop=True;control.should_save=True;status['budget_stop_requested']=True
            def on_log(self,args,state,control,logs=None,**kwargs):append(out/'metrics.jsonl',dict(step=state.global_step,**(logs or {})))
        try:
            trainer=Trainer(model=model,args=config,train_dataset=prepared,data_collator=lambda b:b[0],callbacks=[Audit()])
            result=trainer.train();trainer.save_model(str(out/'final'));trainer.save_state()
            changed=any(not torch.equal(initial[n],w.detach().cpu()) for n,w in model.named_parameters() if n in initial);assert changed
            assert status.get('budget_stop_requested') or trainer.state.global_step==settings['steps']
            status.update(status='budget_stopped' if status.get('budget_stop_requested') else 'completed',step=trainer.state.global_step,
                metrics=result.metrics,weights_changed=changed,adapter_sha256=sha(out/'final/adapter_model.safetensors'),finished_at=stamp())
            write(out/'status.json',status)
        except BaseException as e:
            status.update(status='failed',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc());write(out/'status.json',status);raise

if __name__=='__main__':main()

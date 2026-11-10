"""Assistant JSON SFT with an input-only photometric branch and bounded save/stop."""
import argparse,hashlib,json,os,shutil,signal,traceback
from collections import Counter
from pathlib import Path
from functional_v1.common import ROOT,read,write,append,sha,stamp,GPUJob,BudgetExceeded
from functional_v18.prepare_training import streaming_sha
from functional_v18.objective import sequence_logp
from .runtime import EXP,DATA,ROUTES,verify_origin,expected_count
from .bridge import set_features
from .outcomes import EDITOR_VERSION,SCHEMA_VERSION

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--route',choices=ROUTES,default='photometric_bridge')
    parser.add_argument('--gpu',choices=['0','2','3'],required=True);parser.add_argument('--seed',type=int,default=20261129);args=parser.parse_args()
    protocol=read(EXP/'training-protocol.json');settings=protocol['training']
    assert args.seed in (settings['seed'],settings['replicate_seed'])
    if args.seed==settings['replicate_seed']:
        selection=read(EXP/'development-selection.json');assert selection['qualified'] and selection['route']==args.route
    for name,digest in protocol['code_hashes'].items():assert sha(ROOT/name)==digest,name
    for name,key in [('train-accepted.json','dataset_sha256'),('cases.json','cases_sha256'),('training-pool.json','training_pool_sha256'),('admission.json','admission_sha256')]:
        assert sha(DATA/name)==protocol[key]
    assert sha(EXP/'evaluation-protocol.json')==protocol['evaluation_protocol_sha256']
    assert sha(ROOT/protocol['input_features_path'])==protocol['input_features_sha256']
    assert sha(ROOT/protocol['source_input_audit_path'])==protocol['source_input_audit_sha256']
    assert streaming_sha(ROOT/protocol['base_cache_path'])==protocol['base_cache_sha256'];verify_origin()
    import torch
    torch.set_num_threads(4)
    cache=torch.load(ROOT/protocol['base_cache_path'],map_location='cpu',weights_only=True)
    rows=read(DATA/'train-accepted.json');cases={c['id']:c for c in read(DATA/'cases.json')};features=read(ROOT/protocol['input_features_path'])['cases']
    assert len(cache)==len(rows)==1850
    prepared=[]
    for row in read(DATA/'training-pool.json'):
        index=row['base_row_index'];cid=rows[index]['case_id'];f=features[cid]
        assert f['input_sha256']==cases[cid]['input_sha256'] and f['image']==cases[cid]['image']
        prepared.append(dict(id=row['id'],kind=row['kind'],chosen=cache[index],features=torch.tensor(f['features'],dtype=torch.float32)))
    assert len(prepared)==2002
    out=EXP/'runs'/f'{args.route}-seed-{args.seed}';out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(Path(__file__).parent,out/'source',ignore=shutil.ignore_patterns('__pycache__'));write(out/'protocol-snapshot.json',protocol)
    status=dict(status='starting',pid=os.getpid(),route=args.route,gpu=args.gpu,started_at=stamp());write(out/'status.json',status)
    stop={'requested':False,'signal':None}
    def on_signal(signum,frame):stop.update(requested=True,signal=signum)
    signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
    with GPUJob(args.gpu,'v19-train-'+args.route,max_seconds=3600) as budget:
        with budget.ledger() as value:
            next(j for j in value['jobs'] if j['id']==budget.id).update(editor=EDITOR_VERSION,schema=SCHEMA_VERSION,runtime='functional_v19.runtime')
        from transformers import Trainer,TrainingArguments,TrainerCallback,set_seed
        from .runtime import load_model,processor,predict
        set_seed(args.seed);budget.tick()
        try:
            model=load_model(args.route,train=True,seed=args.seed);write(out/'initialization.json',model.functional_initialization)
            initial={name:p.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad};assert len(initial)==120
            write(out/'initial-adapter-fingerprints.json',{name:hashlib.sha256(value.numpy().tobytes()).hexdigest() for name,value in initial.items()})
            calls=Counter()
            class BridgeTrainer(Trainer):
                def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
                    set_features(model,inputs['features']);batch=inputs['chosen']
                    output=model(**{k:v for k,v in batch.items() if k!='labels'})
                    chosen,count=sequence_logp(output.logits,batch['labels']);loss=-chosen/count
                    assert torch.isfinite(loss);calls[inputs['kind']]+=1
                    return (loss,output) if return_outputs else loss
                def create_optimizer(self):
                    if self.optimizer is None:
                        lora=[p for name,p in model.named_parameters() if p.requires_grad and '.stat_bridge.' not in name]
                        bridge=[p for name,p in model.named_parameters() if p.requires_grad and '.stat_bridge.' in name]
                        assert len(lora)==116 and len(bridge)==4
                        self.optimizer=torch.optim.AdamW([dict(params=lora,lr=settings['learning_rate']),dict(params=bridge,lr=settings['bridge_learning_rate'])],weight_decay=0.)
                        write(out/'optimizer-groups.json',dict(LoRA_tensors=116,LoRA_parameters=sum(p.numel() for p in lora),LoRA_LR=settings['learning_rate'],
                            bridge_tensors=4,bridge_parameters=sum(p.numel() for p in bridge),bridge_LR=settings['bridge_learning_rate']))
                    return self.optimizer
            first=next(r for r in prepared if r['kind']=='preference');set_features(model,first['features'])
            sample={k:v.to('cuda') for k,v in first['chosen'].items()};model.eval()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                checked=model(**sample);logp,count=sequence_logp(checked.logits,sample['labels'])
            assert torch.allclose(-logp/count,checked.loss.float(),atol=1e-4,rtol=1e-4)
            write(out/'real-mask-preflight.json',dict(at=stamp(),passed=True,assistant_tokens=count,independent_CE=float(-logp/count),builtin_CE=float(checked.loss),optimizer_steps=0))
            del checked,sample,logp
            path=ROOT/protocol['parent_diagnostic_predictions_path'];assert sha(path)==protocol['parent_diagnostic_predictions_sha256']
            parent={r['case_id']:r for r in map(json.loads,path.read_text().splitlines())};proc=processor();parity=[]
            for cid in protocol['initial_parity_TRAIN_case_ids']:
                response,decision=predict(model,proc,cases[cid],'en');parity.append(dict(case_id=cid,response=response,decision=decision,same_response_as_reloaded_parent=response==parent[cid]['response']));budget.tick()
            write(out/'initial-function-parity.json',dict(at=stamp(),records=parity,optimizer_steps=0,scope='3 fixed TRAIN prefixes, not universal numeric equivalence'))
            assert all(r['same_response_as_reloaded_parent'] for r in parity),'Zero bridge initial function differed; stop before optimizer'
            model.train()
            config=TrainingArguments(output_dir=str(out),num_train_epochs=2,per_device_train_batch_size=1,gradient_accumulation_steps=4,
                learning_rate=settings['learning_rate'],warmup_ratio=.05,lr_scheduler_type='cosine',optim='adamw_torch',weight_decay=0.,bf16=True,
                gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},max_grad_norm=1.,save_strategy='epoch',logging_steps=1,
                report_to='none',remove_unused_columns=False,seed=args.seed,data_seed=args.seed,dataloader_num_workers=0,disable_tqdm=True)
            write(out/'training-arguments.json',config.to_dict())
            class Audit(TrainerCallback):
                def on_pre_optimizer_step(self,arguments,state,control,**kwargs):
                    active={n:p for n,p in model.named_parameters() if p.requires_grad}
                    assert set(active)==set(initial) and len(active)==120
                    assert all(p.dtype==torch.float32 and p.grad is not None and torch.isfinite(p.grad).all() for p in active.values())
                    if state.global_step==0:
                        assert {id(p) for group in kwargs['optimizer'].param_groups for p in group['params']}=={id(p) for p in active.values()}
                        gradients={name:dict(norm=float(p.grad.norm()),nonzero=int(p.grad.count_nonzero())) for name,p in active.items()}
                        assert all(p.grad.count_nonzero()==0 for name,p in active.items() if '.stat_bridge.' in name and '.first.' in name)
                        assert any(p.grad.count_nonzero()>0 for name,p in active.items() if '.stat_bridge.' in name and '.last.' in name)
                        write(out/'first-gradient-check.json',dict(at=stamp(),finite=True,tensors=120,only_adapter_optimizer=True,initial_bridge_first_layer_zero_expected=True,gradients=gradients))
                def on_step_end(self,arguments,state,control,**kwargs):
                    status.update(status='running',step=state.global_step,epoch=state.epoch,at=stamp());write(out/'status.json',status)
                    try:budget.tick()
                    except BudgetExceeded:stop['requested']=True
                    if stop['requested']:
                        control.should_training_stop=True;control.should_save=True;status.update(budget_stop_requested=True,stop_signal=stop['signal'])
                def on_log(self,arguments,state,control,logs=None,**kwargs):append(out/'metrics.jsonl',dict(step=state.global_step,**(logs or {})))
            trainer=BridgeTrainer(model=model,args=config,train_dataset=prepared,data_collator=lambda batch:batch[0],callbacks=[Audit()])
            trainer.model_accepts_loss_kwargs=False;set_seed(args.seed)
            result=trainer.train();trainer.save_model(str(out/'final'));trainer.save_state()
            if status.get('budget_stop_requested'):
                trainer._save_checkpoint(model,trial=None);write(out/'resume-checkpoint.json',dict(step=trainer.state.global_step,optimizer_scheduler_RNG_saved=True))
            changes={name:not torch.equal(initial[name],p.detach().cpu()) for name,p in model.named_parameters() if name in initial}
            assert all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad)
            assert status.get('budget_stop_requested') or (trainer.state.global_step==1002 and sum(changes.values())==120)
            status.update(status='budget_stopped' if status.get('budget_stop_requested') else 'completed',step=trainer.state.global_step,finished_at=stamp(),metrics=result.metrics,
                changed_language_tensors=sum(v for n,v in changes.items() if 'visual' not in n),changed_merger_tensors=sum(v for n,v in changes.items() if '.merger.' in n),
                changed_bridge_tensors=sum(v for n,v in changes.items() if '.stat_bridge.' in n),adapter_sha256=sha(out/'final/adapter_model.safetensors'),actual_objective_calls=dict(calls))
            if not status.get('budget_stop_requested'):assert calls==dict(anchor=3700,preference=304)
            write(out/'status.json',status)
        except BaseException as error:
            status.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),at=stamp());write(out/'status.json',status);raise

if __name__=='__main__':main()

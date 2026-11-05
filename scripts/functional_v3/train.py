"""Small, matched correction pilots. Dataset admission is required; no held-out training rows."""
import argparse,json,os,shutil,traceback,math
from pathlib import Path
from .common import ROOT,DATA,EXP,PARENT,ROUND2_DATA,GPUJob,read,write,append,sha,stamp


def prepared_rows():
    from functional_v1.runtime import processor,build_batch
    p=processor();cases={c['id']:c for c in read(DATA/'cases.json')};rows=read(DATA/'training-pool.json');prepared=[];audit=[]
    for r in rows:
        c=cases[r['case_id']];assert c['split']=='train' and c['source_id'].startswith('train-')
        assert sha(ROOT/c['image'])==c['input_sha256']
        chosen=build_batch(p,c,r['language'],r['chosen'])
        row={'id':r['id'],'kind':r['kind'],'chosen':chosen}
        if r['kind']=='preference':row['rejected']=build_batch(p,c,r['language'],r['rejected'])
        prepared.append(row);audit.append({'id':r['id'],'kind':r['kind'],'tokens':int(chosen['input_ids'].numel()),'assistant_tokens':int(chosen['labels'].ne(-100).sum()),'image_sha256':c['input_sha256']})
    return prepared,audit


def reference(gpu):
    import torch
    from functional_v1.runtime import load_model
    from .objective import sequence_logp
    cache=DATA/'reference-logps.json';protocol=read(EXP/'training-protocol.json')
    if cache.exists():
        old=read(cache);assert old['pool_sha256']==sha(DATA/'training-pool.json') and old['parent_sha256']==sha(PARENT/'adapter_model.safetensors');return
    with GPUJob(gpu,'v3-precompute-frozen-reference',max_seconds=3600) as budget:
        torch.set_num_threads(4);prepared,audit=prepared_rows();model=load_model(PARENT,train=True);model.eval();rows={}
        for row in prepared:
            if row['kind']!='preference':continue
            value={}
            for side in ('chosen','rejected'):
                budget.tick();batch={k:v.to('cuda') for k,v in row[side].items()};labels=batch.pop('labels')
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    outputs=model(**batch);logp,count=sequence_logp(outputs.logits,labels)
                value[side]=float(logp);value[side+'_tokens']=count
                del outputs,batch,labels
            rows[row['id']]=value
        write(cache,{'at':stamp(),'parent_sha256':sha(PARENT/'adapter_model.safetensors'),'pool_sha256':sha(DATA/'training-pool.json'),
           'cases_sha256':sha(DATA/'cases.json'),'numerics':'Same train=True NF4/fp32 norm+LoRA loader as both optimization routes; eval/no dropout; sequence sums over assistant tokens only','rows':rows})
        write(EXP/'reference-token-audit.json',audit)


def train(route,gpu,seed):
    import torch
    from transformers import Trainer,TrainerCallback,TrainingArguments,set_seed
    from functional_v1.runtime import load_model
    from .objective import sequence_logp,dpo_loss
    protocol=read(EXP/'training-protocol.json');admitted=read(DATA/'admission.json')
    assert protocol['status']=='frozen_before_training'
    assert sha(DATA/'training-pool.json')==protocol['pool_sha256']==admitted['pool_sha256']
    assert sha(DATA/'cases.json')==protocol['cases_sha256']
    assert sha(PARENT/'adapter_model.safetensors')==protocol['parent_sha256']
    assert sha(DATA/'admission.json')==protocol['admission_sha256']
    for file,digest in protocol['core_code_hashes'].items():assert sha(ROOT/file)==digest
    ref=read(DATA/'reference-logps.json');assert ref['pool_sha256']==protocol['pool_sha256'] and ref['parent_sha256']==protocol['parent_sha256']
    out=EXP/'runs'/f'{route}-seed-{seed}';out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(Path(__file__).parent,out/'source',ignore=shutil.ignore_patterns('__pycache__'))
    write(out/'protocol-snapshot.json',protocol);write(out/'settings.json',{'route':route,'seed':seed,**protocol['training'],'reference_cache_sha256':sha(DATA/'reference-logps.json')})
    status={'status':'starting','pid':os.getpid(),'gpu':gpu,'at':stamp()};write(out/'status.json',status)
    with GPUJob(gpu,'v3-train-'+route,max_seconds=5400) as budget:
        torch.set_num_threads(4);set_seed(seed);prepared,audit=prepared_rows();write(out/'token-audit.json',audit)
        for row in prepared:
            if row['kind']=='preference':row['reference']=ref['rows'][row['id']]
        model=load_model(PARENT,train=True)
        initial={n:w.detach().cpu().clone() for n,w in model.named_parameters() if w.requires_grad}
        class CorrectionTrainer(Trainer):
            def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
                chosen_batch=inputs['chosen'];labels=chosen_batch['labels']
                output=model(**{k:v for k,v in chosen_batch.items() if k!='labels'})
                chosen,count=sequence_logp(output.logits,labels)
                nll=-chosen/count
                if route=='sft' or inputs['kind']=='anchor':loss=nll
                else:
                    rejected_batch=inputs['rejected'];rejected_output=model(**{k:v for k,v in rejected_batch.items() if k!='labels'})
                    rejected,_=sequence_logp(rejected_output.logits,rejected_batch['labels'])
                    frozen=inputs['reference'];loss=dpo_loss(chosen,rejected,frozen['chosen'],frozen['rejected'],protocol['training']['beta'])+protocol['training']['chosen_sft_weight_on_preference_rows']*nll
                return (loss,output) if return_outputs else loss
        # Same real batch checked against the model's independent built-in causal CE before updates.
        sample={k:v.to('cuda') for k,v in prepared[0]['chosen'].items()};model.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            result=model(**sample);logp,count=sequence_logp(result.logits,sample['labels'])
        assert torch.allclose(-logp/count,result.loss.float(),rtol=1e-4,atol=1e-4)
        if prepared[0]['kind']=='preference':
            assert abs(float(logp)-ref['rows'][prepared[0]['id']]['chosen'])<1e-4, 'Frozen reference and initial policy numerics disagree'
        write(out/'real-mask-preflight.json',{'passed':True,'assistant_tokens':count,'assistant_nll':float(-logp/count),'builtin_causal_CE':float(result.loss),'comparison':'Independent built-in loss with the same prefix mask, before any parameter update.'})
        del result,sample,logp;model.train()
        args=TrainingArguments(output_dir=str(out),num_train_epochs=2,per_device_train_batch_size=1,
           gradient_accumulation_steps=4,learning_rate=1e-5,warmup_ratio=.05,lr_scheduler_type='cosine',
           optim='adamw_torch',weight_decay=0.,bf16=True,gradient_checkpointing=True,
           gradient_checkpointing_kwargs={'use_reentrant':False},max_grad_norm=1.,save_strategy='epoch',
           logging_steps=1,report_to='none',remove_unused_columns=False,seed=seed,data_seed=seed,
           dataloader_num_workers=0,disable_tqdm=True)
        write(out/'training-arguments.json',args.to_dict())
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
            trainer=CorrectionTrainer(model=model,args=args,train_dataset=prepared,data_collator=lambda rows:rows[0],callbacks=[Audit()])
            trainer.model_accepts_loss_kwargs=False
            result=trainer.train();trainer.save_model(str(out/'final'));trainer.save_state()
            changed=any(not torch.equal(initial[n],w.detach().cpu()) for n,w in model.named_parameters() if n in initial);assert changed
            status.update(status='budget_stopped' if status.get('budget_stop_requested') else 'completed',step=trainer.state.global_step,
              expected_steps=2*math.ceil(len(prepared)/4),weights_changed=changed,adapter_sha256=sha(out/'final/adapter_model.safetensors'),metrics=result.metrics,finished_at=stamp())
            assert status.get('budget_stop_requested') or trainer.state.global_step==status['expected_steps']
            write(out/'status.json',status)
        except BaseException as error:
            status.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc());write(out/'status.json',status);raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['reference','train']);ap.add_argument('--route',choices=['dpo','sft']);ap.add_argument('--gpu',choices=['0','2','3'],required=True);ap.add_argument('--seed',type=int,default=20261107);a=ap.parse_args()
    if a.command=='reference':reference(a.gpu)
    else:train(a.route,a.gpu,a.seed)
if __name__=='__main__':main()

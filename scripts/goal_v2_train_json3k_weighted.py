"""Bounded512-update train-only quality-weighted loss control;1024-step schedule preserved."""
import argparse, hashlib, json, math, os, shutil, subprocess, time, traceback
from pathlib import Path
import torch
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
from trl.models.utils import peft_module_casting_to_bf16
from stage2_runtime import ROOT,load_model,processor,prompt,image_input,adapter_state,tensor_digest,assert_trainable
from train_stage2_sft_control import build_batch
from goal_v2_json_runtime import prepare_row
def write(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
class QualityWeightedTrainer(Trainer):
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        weight=inputs['quality_weight']
        model_inputs={k:v for k,v in inputs.items() if k!='quality_weight'}
        loss,outputs=super().compute_loss(model,model_inputs,return_outputs=True,num_items_in_batch=num_items_in_batch)
        weighted=loss*weight
        return (weighted,outputs) if return_outputs else weighted

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--mode',choices=['full','parameters'],required=True)
    ap.add_argument('--lr',type=float,required=True);ap.add_argument('--gpu',required=True);ap.add_argument('--seed',type=int,default=20261103)
    ap.add_argument('--training-file',choices=['train3k.json','train3k-contrast-calibrated.json'],default='train3k-contrast-calibrated.json')
    a=ap.parse_args();assert a.gpu in ['0','2','3'] and os.environ['CUDA_VISIBLE_DEVICES']==a.gpu
    assert int(subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))<512
    out=ROOT/'outputs/goal-v2-20261103'/a.name;out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,out/'training-source.py')
    shutil.copyfile(ROOT/'scripts/goal_v2_json_runtime.py',out/'json-runtime-source.py')
    write(out/'prompt-variant.json',{'variant':'json-only-schema-v1','runtime_sha256':hashlib.sha256((ROOT/'scripts/goal_v2_json_runtime.py').read_bytes()).hexdigest()})
    start=time.monotonic();status={'status':'loading','pid':os.getpid(),'gpu':a.gpu,'step':0,'settings':vars(a)};write(out/'status.json',status)
    monitor=subprocess.Popen(['nvidia-smi','-i',a.gpu,'--query-gpu=timestamp,memory.used,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits','--loop=5'],stdout=(out/'gpu.csv').open('w'))
    try:
        set_seed(a.seed);torch.set_num_threads(4)
        data=ROOT/'data/goal-v2-20261103';manifest=json.loads((data/'training-variants-3k.json').read_text());source=data/a.training_file
        entry=manifest['files']['train3k']
        if a.training_file!='train3k.json':
            assert a.mode=='parameters', 'Calibrated parameters require reasoning tokens to be excluded from loss'
            variants=json.loads((data/'training-variants-3k.json').read_text())
            entry=variants['files']['train3k-contrast-calibrated']
            assert entry['source_sha256']==manifest['files']['train3k']['sha256']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==entry['sha256']
        write(out/'training-input.json',entry)
        quality_source=data/'train3k-quality-weights.json';quality=json.loads(quality_source.read_text());assert quality['source_sha256']==entry['sha256']
        quality_lookup={r['id']:r for r in quality['rows']};assert len(quality_lookup)==3000
        write(out/'quality-weight-identity.json',{'source_sha256':hashlib.sha256(quality_source.read_bytes()).hexdigest(),'formula':quality['formula'],'all_rows_retained':True})
        rows=json.loads(source.read_text());assert len(rows)==3000;p=processor();prepared=[];audit=[]
        for row in rows:
            row=prepare_row(row,training=True)
            batch=build_batch(p,row)
            if a.mode=='parameters':
                target=next(m['content'] for m in row['conversations'] if m['role']=='assistant')
                assert target.count('```json')==1
                text=p.apply_chat_template(prompt(row)+[{'role':'assistant','content':target}],tokenize=False,add_generation_prompt=False)
                cut=text.rindex('```json')
                prefix=p(text=[text[:cut]],images=[image_input(row['images'][0])],return_tensors='pt',add_special_tokens=False)['input_ids']
                n=prefix.shape[1]
                # BPE can join the final whitespace with the first fence token.
                while n>0 and not torch.equal(batch['input_ids'][:,:n],prefix[:,:n]): n-=1
                assert n>=prefix.shape[1]-4 and n>0
                batch['labels'][:,:n]=-100
            assert quality_lookup[row['id']]['group_id']==row['group_id'];weight=quality_lookup[row['id']]['weight'];assert weight>0 and math.isfinite(weight)
            batch['quality_weight']=torch.tensor(weight,dtype=torch.float32)
            count=int((batch['labels']!=-100).sum());assert count>5
            audit.append({'id':row['id'],'input_tokens':int(batch['input_ids'].numel()),'supervised_tokens':count})
            prepared.append(batch)
        write(out/'token-audit.json',{'rows':audit,'total_supervised_tokens':sum(r['supervised_tokens'] for r in audit),'training_sha256':entry['sha256']})
        model=load_model();peft_module_casting_to_bf16(model);initial=tensor_digest(adapter_state(model))
        config=TrainingArguments(output_dir=str(out),max_steps=1024,per_device_train_batch_size=1,gradient_accumulation_steps=4,
            learning_rate=a.lr,warmup_ratio=.05,lr_scheduler_type='cosine',optim='adamw_torch',weight_decay=0.,bf16=True,
            gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},max_grad_norm=1.,
            save_strategy='steps',save_steps=256,logging_steps=1,report_to='none',remove_unused_columns=False,
            seed=a.seed,data_seed=a.seed,dataloader_num_workers=0,disable_tqdm=True)
        write(out/'run-config.json',config.to_dict())
        write(out/'training-scope.json',{'rows':3000,'epochs':2048/3000,'expected_sample_exposures':2048,'steps':512,'scheduler_horizon':1024,'checkpoint_validation_steps':[512],'purpose':'Longer-optimization control; full3k training, no held-out test access'})
        class Audit(TrainerCallback):
            def on_pre_optimizer_step(self,args,state,control,**kwargs):
                assert_trainable(model)
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad)
            def on_step_end(self,args,state,control,**kwargs):
                status.update(status='running',step=state.global_step,elapsed_seconds=time.monotonic()-start);write(out/'status.json',status)
                if state.global_step==512:control.should_training_stop=True;control.should_save=True
                if time.monotonic()-start>7200:control.should_training_stop=True;control.should_save=True
            def on_log(self,args,state,control,logs=None,**kwargs):
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':state.global_step,**(logs or {})},allow_nan=False)+'\n')
        trainer=QualityWeightedTrainer(model=model,args=config,train_dataset=prepared,data_collator=lambda b:b[0],callbacks=[Audit()])
        result=trainer.train();assert trainer.state.global_step==512 and abs(trainer.state.epoch-2048/3000)<1e-6;trainer.save_model(str(out/'final'));trainer.save_state()
        assert tensor_digest(adapter_state(model))!=initial
        status.update(status='completed' if trainer.state.global_step==512 else 'time_limit_stopped',metrics=result.metrics)
        write(out/'final-identity.json',{'sha256':hashlib.sha256((out/'final/adapter_model.safetensors').read_bytes()).hexdigest(),'steps':trainer.state.global_step})
    except BaseException as exc:
        status.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        status['elapsed_seconds']=time.monotonic()-start;write(out/'status.json',status);monitor.terminate();monitor.wait(timeout=10)
if __name__=='__main__':main()

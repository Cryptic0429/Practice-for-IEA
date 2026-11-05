"""Measure actual LLaMA-Factory QLoRA updates on training-only pilot samples."""
import argparse
import copy
import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import yaml
from transformers import TrainerCallback, set_seed
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer, SFTDataCollatorWith4DAttentionMask
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

ROOT=Path('/mnt/sdc/zhc/IEA')


class Monitor:
    def __init__(self):
        self.gpu=os.environ.get('CUDA_VISIBLE_DEVICES','2')
        assert ',' not in self.gpu
        baseline=subprocess.check_output(['nvidia-smi','-i',self.gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if int(baseline.strip())>512: raise RuntimeError('Selected GPU is occupied')
        self.phase='setup'
        self.rows=[]
        self.proc=subprocess.Popen(['nvidia-smi','-i',self.gpu,'--query-gpu=memory.used,memory.total,utilization.gpu',
                                    '--format=csv,noheader,nounits','--loop-ms=200'],stdout=subprocess.PIPE,text=True)
        self.thread=threading.Thread(target=self.read,daemon=True)
        self.thread.start()
    def read(self):
        for line in self.proc.stdout:
            try:
                used,total,util=[int(v.strip()) for v in line.split(',')]
                self.rows.append(dict(time=time.time(),phase=self.phase,used_mib=used,total_mib=total,utilization=util))
            except ValueError: pass
    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.thread.join(timeout=2)


class MeasuredTrainer(CustomSeq2SeqTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.samples=0
        self.input_tokens=0
        self.target_tokens=0
        self.micro_losses=[]
    def training_step(self,model,inputs,num_items_in_batch=None):
        self.samples+=int(inputs['input_ids'].shape[0])
        self.input_tokens+=int(inputs['attention_mask'].sum().item())
        self.target_tokens+=int((inputs['labels']!=-100).sum().item())
        loss=super().training_step(model,inputs,num_items_in_batch)
        self.micro_losses.append(loss.detach())
        return loss


class Timer(TrainerCallback):
    def __init__(self,trainer,monitor,warmup=2):
        self.trainer=trainer
        self.monitor=monitor
        self.warmup=warmup
        self.steps=[]
        self.window=None
        self.last_samples=0
    def on_train_begin(self,args,state,control,**kwargs):
        torch.cuda.synchronize()
        self.monitor.phase='training'
        self.load_peak_allocated=torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        self.started=time.perf_counter()
        self.last=self.started
        if self.warmup==0:
            self.window=(self.started,0,0,0)
    def on_step_end(self,args,state,control,**kwargs):
        torch.cuda.synchronize()
        now=time.perf_counter()
        losses=torch.stack(self.trainer.micro_losses).float().cpu()
        assert torch.isfinite(losses).all(), 'Non-finite training loss'
        self.trainer.micro_losses.clear()
        self.steps.append(dict(step=state.global_step,seconds=now-self.last,
                               samples=self.trainer.samples-self.last_samples,
                               accumulated_loss=float(losses.sum()),
                               max_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                               max_reserved_gib=torch.cuda.max_memory_reserved()/1024**3))
        if state.global_step==self.warmup:
            self.window=(now,self.trainer.samples,self.trainer.input_tokens,self.trainer.target_tokens)
        self.last_samples=self.trainer.samples
        self.last=time.perf_counter()
    def on_train_end(self,args,state,control,**kwargs):
        torch.cuda.synchronize()
        self.ended=time.perf_counter()
        self.peak_allocated=torch.cuda.max_memory_allocated()
        self.peak_reserved=torch.cuda.max_memory_reserved()
        self.monitor.phase='post_training'


def arguments(name,pixels,rank,targets,batch,accum,steps):
    return dict(model_name_or_path=str(ROOT/'models/Qwen2.5-VL-7B-Instruct'),trust_remote_code=False,
                quantization_bit=4,quantization_method='bnb',quantization_type='nf4',double_quantization=True,
                image_max_pixels=pixels,stage='sft',do_train=True,finetuning_type='lora',
                lora_rank=rank,lora_alpha=rank*2,lora_target=targets,lora_dropout=0.0,
                freeze_vision_tower=True,freeze_multi_modal_projector=True,flash_attn='sdpa',
                dataset='iea_edit_3k_probe_train',dataset_dir=str(ROOT/'data'),template='qwen2_vl',
                cutoff_len=2048,preprocessing_num_workers=2,dataloader_num_workers=2,
                output_dir=str(ROOT/'outputs/benchmarks'/name),report_to='none',logging_steps=1,
                per_device_train_batch_size=batch,gradient_accumulation_steps=accum,max_steps=steps,
                learning_rate=1e-4,lr_scheduler_type='constant',warmup_ratio=0,
                bf16=True,gradient_checkpointing=True,upcast_layernorm=True,optim='adamw_torch',
                save_strategy='no',eval_strategy='no',seed=20261031,data_seed=20261031,
                remove_unused_columns=False,overwrite_output_dir=False,disable_tqdm=True)


def eval_probe(model,collator,dataset):
    model.eval()
    batch=collator([dataset[0]])
    batch={k:v.to(model.device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): output=model(**batch)
    return dict(loss=float(output.loss),logits=output.logits[0,-2,:32].float().cpu().tolist(),
                input_ids_sha256=hashlib.sha256(json.dumps(dataset[0]['input_ids']).encode()).hexdigest())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',required=True)
    parser.add_argument('--pixels',type=int,required=True)
    parser.add_argument('--rank',type=int,default=8)
    parser.add_argument('--targets',default='q_proj,v_proj')
    parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--accum',type=int,default=16)
    parser.add_argument('--steps',type=int,default=12)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--dataset',choices=['iea_edit_3k_probe_train','iea_edit_3k_stress_train'],default='iea_edit_3k_probe_train')
    cli=parser.parse_args()
    assert 0 <= cli.warmup < cli.steps
    config=arguments(cli.name,cli.pixels,cli.rank,cli.targets,cli.batch,cli.accum,cli.steps)
    config['dataset']=cli.dataset
    out=Path(config['output_dir'])
    if out.exists() and any(out.iterdir()): raise RuntimeError('Benchmark output already exists')
    out.mkdir(parents=True,exist_ok=True)
    (out/'run-config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    started=time.perf_counter()
    monitor=Monitor()
    try:
        ma,da,ta,fa,ga=get_train_args(config)
        set_seed(ta.seed)
        tm=load_tokenizer(ma)
        template=get_template_and_fix_tokenizer(tm['tokenizer'],da)
        dm=get_dataset(template,ma,da,ta,stage='sft',**tm)
        ds=dm['train_dataset']
        registry=json.loads((ROOT/'data/dataset_info.json').read_text())
        raw=json.loads(Path(registry[cli.dataset]['file_name']).read_text())
        assert len(ds)==len(raw)
        full_da=copy.deepcopy(da)
        full_da.cutoff_len=32768
        proc=SupervisedDatasetProcessor(template=template,data_args=full_da,**tm)
        lengths=[]
        for i,row in enumerate(raw):
            c=row['conversations']
            ids,labels=proc._encode_data_example([c[1]],[c[2]],c[0]['content'],None,row['images'],[],[])
            assert ids==ds[i]['input_ids'] and labels==ds[i]['labels'], f'Truncation in {row["id"]}'
            assert any(x!=-100 for x in labels) and any(x==-100 for x in labels)
            lengths.append(len(ids))
        model=load_model(tm['tokenizer'],ma,fa,is_trainable=True)
        trainable={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        assert trainable and all('lora_' in n and 'visual' not in n for n in trainable)
        collator=SFTDataCollatorWith4DAttentionMask(template=template,model=model,pad_to_multiple_of=8,
                    label_pad_token_id=-100,attn_implementation='sdpa',compute_dtype=ma.compute_dtype,**tm)
        trainer=MeasuredTrainer(model=model,args=ta,finetuning_args=fa,data_collator=collator,train_dataset=ds,**tm)
        timer=Timer(trainer,monitor,warmup=cli.warmup)
        trainer.add_callback(timer)
        setup_seconds=time.perf_counter()-started
        result=trainer.train()
        assert trainer.state.global_step==cli.steps
        assert timer.window is not None
        delta={n:float((p.detach().cpu()-trainable[n]).abs().max()) for n,p in model.named_parameters() if n in trainable}
        assert max(delta.values())>0, 'Adapters were not updated'
        trainer.save_model()
        trainer.save_state()
        saved_probe=eval_probe(model,collator,ds)
        start,n,t,y=timer.window
        measured_seconds=timer.ended-start
        measured_samples=trainer.samples-n
        sampled=[r for r in monitor.rows if r['phase']=='training']
        steady_steps=timer.steps[timer.warmup:]
        report=dict(timestamp=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),config=config,
                    purpose='longest_training_records_stress' if cli.dataset.endswith('stress_train') else 'throughput',
                    model_revision=json.loads((ROOT/'configs/qwen-base-revision.json').read_text())['revision'],
                    frozen_split_manifest_sha256=hashlib.sha256((ROOT/'data/stage1/splits/image-edit-3k-v1/manifest.json').read_bytes()).hexdigest(),
                    actual_optimizer_steps=trainer.state.global_step,actual_training_samples=trainer.samples,
                    warmup_optimizer_steps=timer.warmup,measured_optimizer_steps=cli.steps-timer.warmup,
                    measured_samples=measured_samples,measured_seconds=measured_seconds,
                    samples_per_second=measured_samples/measured_seconds,
                    input_tokens_per_second=(trainer.input_tokens-t)/measured_seconds,
                    supervised_tokens_per_second=(trainer.target_tokens-y)/measured_seconds,
                    step_seconds_median=statistics.median(s['seconds'] for s in steady_steps),
                    training_peak_allocated_gib=timer.peak_allocated/1024**3,
                    training_peak_reserved_gib=timer.peak_reserved/1024**3,
                    training_peak_device_mib=max(r['used_mib'] for r in sampled),
                    device_total_mib=sampled[0]['total_mib'],
                    median_gpu_utilization=statistics.median(r['utilization'] for r in sampled),
                    setup_seconds=setup_seconds,train_wall_seconds=timer.ended-timer.started,
                    total_wall_seconds=time.perf_counter()-started,
                    trainable_parameters=sum(t.numel() for t in trainable.values()),
                    maximum_adapter_parameter_change=max(delta.values()),changed_adapter_tensors=sum(v>0 for v in delta.values()),
                    vision_trainable=False,all_training_losses_finite=True,all_probe_targets_untruncated=True,
                    probe_sequence_max=max(lengths),probe_sequence_mean=statistics.mean(lengths),
                    saved_adapter_probe=saved_probe,steps=timer.steps,trainer_metrics=result.metrics,
                    notes=['Training-only benchmark; no validation/test model evaluation.',
                           'Timings include data loading, forward, backward, gradient accumulation and optimizer updates after warmup.',
                           'Pilot adapter must not be used as the starting point of formal training.'])
        (out/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
        (ROOT/'logs'/f'{cli.name}-benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
        print('BENCHMARK_PASSED',json.dumps({k:report[k] for k in ['samples_per_second','training_peak_allocated_gib','training_peak_reserved_gib','training_peak_device_mib','actual_optimizer_steps','measured_samples','maximum_adapter_parameter_change']}),flush=True)
    finally:
        monitor.close()
        (out/'gpu-monitor.json').write_text(json.dumps(monitor.rows,indent=2)+'\n')


if __name__=='__main__':main()

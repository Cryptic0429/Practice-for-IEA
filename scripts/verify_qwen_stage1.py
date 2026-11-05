"""Readiness check only: real SFT inputs and finite forward loss, no training."""
import copy
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer, SFTDataCollatorWith4DAttentionMask
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor

ROOT=Path('/mnt/sdc/zhc/IEA')


def main():
    args=dict(model_name_or_path=str(ROOT/'models/Qwen2.5-VL-7B-Instruct'),trust_remote_code=False,
              quantization_bit=4,quantization_method='bnb',image_max_pixels=65536,
              stage='sft',do_train=True,finetuning_type='lora',lora_rank=8,lora_target='q_proj,v_proj',
              freeze_vision_tower=True,freeze_multi_modal_projector=True,flash_attn='sdpa',
              dataset='gier_sft_distill,gier_sft_summary',dataset_dir=str(ROOT/'data'),template='qwen2_vl',
              cutoff_len=2048,max_samples=4,preprocessing_num_workers=1,dataloader_num_workers=0,
              output_dir=str(ROOT/'outputs/readiness-forward-only'),report_to='none',
              per_device_train_batch_size=1,gradient_accumulation_steps=1,bf16=True,
              gradient_checkpointing=True,upcast_layernorm=True)
    ma,da,ta,fa,ga=get_train_args(args)
    tm=load_tokenizer(ma)
    template=get_template_and_fix_tokenizer(tm['tokenizer'],da)
    dm=get_dataset(template,ma,da,ta,stage='sft',**tm)
    ds=dm['train_dataset']
    assert len(ds)==8
    long_da=copy.deepcopy(da)
    long_da.cutoff_len=32768
    full_processor=SupervisedDatasetProcessor(template=template,tokenizer=tm['tokenizer'],processor=tm['processor'],data_args=long_da)
    checks=[]
    for task_index,task in enumerate(['GIER-Edit','GIER-Summary']):
        rows=json.loads((ROOT/'data/stage1/ready'/task/'sft_train.json').read_text())[:4]
        for index,row in enumerate(rows):
            messages=row['conversations']
            ids,labels=full_processor._encode_data_example(
                prompt=[messages[1]],response=[messages[2]],system=messages[0]['content'],tools=None,
                images=row['images'],videos=[],audios=[])
            actual=ds[task_index*4+index]
            assert actual['input_ids']==ids and actual['labels']==labels, 'A real target or prompt was truncated/misaligned'
            supervised=sum(x!=-100 for x in labels)
            assert 0<supervised<len(labels)
            checks.append(dict(task=task,id=row['id'],tokens=len(ids),supervised_tokens=supervised,truncated=False))
    model=load_model(tm['tokenizer'],ma,fa,is_trainable=True)
    model.eval()
    trainable=[(n,p.numel()) for n,p in model.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in n and 'visual' not in n for n,_ in trainable)
    collator=SFTDataCollatorWith4DAttentionMask(template=template,model=model,pad_to_multiple_of=8,
            label_pad_token_id=-100,attn_implementation='sdpa',compute_dtype=ma.compute_dtype,**tm)
    losses=[]
    torch.cuda.reset_peak_memory_stats()
    for idx in [0,4]:
        batch=collator([ds[idx]])
        batch={k:v.to(model.device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            result=model(**batch)
        assert torch.isfinite(result.loss)
        losses.append(dict(task=checks[idx]['task'],loss=float(result.loss)))
    report=dict(timestamp=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),
                original_qwen_loaded=True,quantization_bit=4,real_records_preprocessed=len(ds),samples=checks,
                finite_forward_losses=losses,trainable_adapter_parameters=sum(n for _,n in trainable),
                trainable_parameter_names=[n for n,_ in trainable],vision_trainable=False,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                image_max_pixels=65536,cutoff_len=2048,training_steps=0,
                note='Forward-only readiness check. No optimizer steps, no quality evaluation, no trained adapter produced.')
    (ROOT/'logs/qwen-stage1-readiness.json').write_text(json.dumps(report,indent=2)+'\n')
    print('QWEN_STAGE1_READINESS_PASSED',json.dumps({k:v for k,v in report.items() if k!='trainable_parameter_names'}),flush=True)


if __name__=='__main__':main()

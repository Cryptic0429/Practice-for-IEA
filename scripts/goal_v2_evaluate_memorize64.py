"""JSON-only deterministic generation with matched prompt controls and original frozen editor."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback
import torch
from transformers import set_seed
from trl.models.utils import peft_module_casting_to_bf16
from stage2_runtime import ROOT, ADAPTER, load_model, processor, encoded
from goal_v2_json_runtime import prepare_row
from evaluation_core import Scorer, parse_parameters, sample_seed


def write(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--adapter',required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--split',choices=['train_probe'],required=True)
    parser.add_argument('--gpu',choices=['0','2','3'],required=True)
    parser.add_argument('--max-gpu-seconds',type=float,required=True)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==args.gpu
    assert args.max_gpu_seconds>0
    assert int(subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=memory.used',
                                      '--format=csv,noheader,nounits'],text=True))<512
    adapter=Path(args.adapter)
    identity=dict(adapter=str(adapter),adapter_sha256=sha(adapter/'adapter_model.safetensors'),
                  split=args.split,name=args.name,gpu=args.gpu,seed=20261101)
    assert args.split=='train_probe', 'Training probe only'
    data=ROOT/'data/goal-v2-20261103'
    manifest=json.loads((data/'memorize64-probe-manifest.json').read_text())
    source=data/'train-memorize64.json'
    assert sha(source)==manifest['files'][args.split]['sha256']
    identity['source_sha256']=sha(source)
    if args.split.startswith('test_'):
        lock=json.loads((ROOT/'configs/stage2-selected-models.json').read_text())
        assert lock['status']=='frozen'
        assert identity['adapter_sha256']==lock['models'][args.name]['adapter_sha256']
        assert sha(adapter/'adapter_config.json')==lock['models'][args.name]['adapter_config_sha256']
        for path,digest in lock['evaluation_identity'].items():
            assert sha(path)==digest,path
    protocol=json.loads((ROOT/'configs/evaluation-protocol-v1.json').read_text())
    for path,digest in protocol['code_hashes'].items():
        assert sha(path)==digest
    for path in ['goal_v2_evaluate_memorize64.py','stage2_runtime.py','evaluation_core.py','goal_v2_json_runtime.py']:
        identity[path]=sha(ROOT/'scripts'/path)
    out=ROOT/'outputs/goal-v2-20261103/training-probe-evaluation'/f'{args.split}-{args.name}'
    out.mkdir(parents=True,exist_ok=False)
    worker=(out/'worker.lock').open('w')
    fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic()
    run=dict(identity=identity,pid=os.getpid(),started_at=stamp(),max_gpu_seconds=args.max_gpu_seconds)
    write(out/'run.json',run)
    status=dict(status='loading',pid=os.getpid(),gpu=args.gpu,started_at=run['started_at'],
                updated_at=stamp(),completed=0)
    write(out/'status.json',status)
    monitor=subprocess.Popen(['nvidia-smi','-i',args.gpu,'--query-gpu=timestamp,memory.used,temperature.gpu,utilization.gpu',
                               '--format=csv,noheader,nounits','--loop=2'],stdout=(out/'gpu.csv').open('w'))
    records=[]
    try:
        torch.set_num_threads(4)
        rows=json.loads(source.read_text())
        scorer=Scorer(ROOT,protocol)
        processing=processor()
        model=load_model(adapter,trainable=True)
        peft_module_casting_to_bf16(model)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        for row in rows:
            row=prepare_row(row,training=False)
            if time.monotonic()-started>args.max_gpu_seconds:
                status['status']='budget_stopped'
                break
            set_seed(sample_seed(row['id'],20261101))
            record=dict(id=row['id'],group_id=row['group_id'],model=args.name,
                        instruction=row['extra_info']['instruction'],instruction_type=row['extra_info']['inst_type'],
                        input_image=row['images'][0],reference_image=row['extra_info']['reference_image_path'],
                        inference_error=None)
            begin=time.monotonic()
            try:
                inputs={k:v.to('cuda') for k,v in encoded(processing,row).items()}
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model.generate(**inputs,do_sample=False,max_new_tokens=512,
                        repetition_penalty=1.,pad_token_id=processing.tokenizer.pad_token_id,
                        eos_token_id=processing.tokenizer.eos_token_id,use_cache=True,disable_compile=True)
                tokens=output[0,inputs['input_ids'].shape[1]:].tolist()
                response=processing.decode(tokens,skip_special_tokens=True)
                record.update(response_text=response,response_length=len(tokens),
                              length_limit_reached=len(tokens)>=512,
                              finish_reason='length' if len(tokens)>=512 and processing.tokenizer.eos_token_id not in tokens else 'stop')
            except Exception as exc:
                record.update(response_text='',response_length=0,finish_reason='error',
                              inference_error=f'{type(exc).__name__}:{exc}',traceback=traceback.format_exc())
            record['generation_seconds']=time.monotonic()-begin
            parsed=parse_parameters(record['response_text'],protocol)
            record['parsing']=parsed
            scored=scorer.score(row,parsed,out/'edited'/f'{hashlib.sha256(row["id"].encode()).hexdigest()[:20]}.png')
            record['scoring']=scored
            record['empty_parameters']=parsed.get('parameters')=={}
            record['all_zero_parameters']=parsed.get('parameters') is not None and all(v==0 for v in parsed['parameters'].values())
            record['completed_at']=stamp()
            with (out/'predictions.jsonl').open('a') as f:
                f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            records.append(record)
            if scored.get('metric_error'):
                raise RuntimeError(scored['metric_error'])
            if record['inference_error']:
                raise RuntimeError(record['inference_error'])
            status.update(status='running',completed=len(records),updated_at=stamp(),
                          elapsed_seconds=time.monotonic()-started)
            write(out/'status.json',status)
        if len(records)==len(rows):
            summary=dict(rows=len(records),source_sha256=identity['source_sha256'],
                         adapter_sha256=identity['adapter_sha256'],
                         mean_l1=sum(r['scoring']['edited_metrics']['pixel_L1'] for r in records)/len(records),
                         mean_rmse=sum(r['scoring']['edited_metrics']['pixel_L2'] for r in records)/len(records),
                         baseline_l1=sum(r['scoring']['baseline_metrics']['pixel_L1'] for r in records)/len(records),
                         executable=sum(r['scoring']['editor_success'] for r in records),
                         empty=sum(r['empty_parameters'] for r in records),
                         all_zero=sum(r['all_zero_parameters'] for r in records),
                         truncated=sum(r['finish_reason']=='length' for r in records))
            assert math.isfinite(summary['mean_l1'])
            write(out/'summary.json',summary)
            status['status']='completed'
    except BaseException as exc:
        status.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        status.update(updated_at=stamp(),elapsed_seconds=time.monotonic()-started,
                      gpu_hours=(time.monotonic()-started)/3600,completed=len(records))
        write(out/'status.json',status)
        monitor.terminate()
        monitor.wait(timeout=10)


if __name__=='__main__':
    main()

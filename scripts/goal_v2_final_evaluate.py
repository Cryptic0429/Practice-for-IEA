"""Locked independent71 evaluation. No model selection, content retries or test-time tuning."""
import argparse,datetime,fcntl,hashlib,json,math,os,shutil,subprocess,time,traceback
from pathlib import Path
import torch
from transformers import set_seed
from trl.models.utils import peft_module_casting_to_bf16
from stage2_runtime import ROOT,load_model,processor,encoded
from goal_v2_json_runtime import prepare_row
from evaluation_core import Scorer,parse_parameters,sample_seed
OUT=ROOT/'outputs/goal-v2-20261103'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def clean(row,mode):
    r={'images':list(row['images']),'conversations':[dict(m) for m in row['conversations'] if m['role'] in ('system','user')],
       'extra_info':{'origin_image_path':row['images'][0]}}
    assert len(r['conversations'])==2
    return prepare_row(r,training=False) if mode=='json' else r
def summary_of(records,entry,source,lockfile,model,predictions):
    assert len(records)==71
    return {'rows':71,'model':model,'adapter_sha256':entry['adapter_sha256'],'source_sha256':sha(source),'model_lock_sha256':sha(lockfile),
            'mean_l1':sum(r['scoring']['edited_metrics']['pixel_L1'] for r in records)/71,'mean_rmse':sum(r['scoring']['edited_metrics']['pixel_L2'] for r in records)/71,
            'baseline_l1':sum(r['scoring']['baseline_metrics']['pixel_L1'] for r in records)/71,'executable':sum(r['scoring']['editor_success'] for r in records),
            'empty':sum(r['empty_parameters'] for r in records),'all_zero':sum(r['all_zero_parameters'] for r in records),'truncated':sum(r['finish_reason']=='length' for r in records),'predictions_sha256':sha(predictions)}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--gpu',choices=['0','2','3'],required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--max-gpu-seconds',type=float,default=5400);a=ap.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.gpu and a.max_gpu_seconds>0
    assert int(subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))<512
    lockfile=OUT/'final-models-lock.json';lock=json.loads(lockfile.read_text());assert lock['status']=='frozen_for_final_test'
    protocolfile=OUT/'final-protocol.json';final_protocol=json.loads(protocolfile.read_text());assert final_protocol['status']=='frozen_before_test_inference' and sha(protocolfile)==lock['final_protocol_sha256']
    assert datetime.datetime.fromisoformat(lock['locked_at'])<=datetime.datetime.now(datetime.timezone.utc)
    gatefile=OUT/'multiseed-development-gate.json';gate=json.loads(gatefile.read_text());assert gate['passed'] and sha(gatefile)==lock['development_gate_sha256']
    assert len(lock['candidate_keys'])==3 and sorted(lock['training_seeds'])==[20261103,20261104,20261105]
    assert set(lock['baseline_keys'])=={'A','B','SFT','GRPO'} and a.model in lock['models']
    for path,digest in lock['code_hashes'].items():assert sha(ROOT/path)==digest,path
    entry=lock['models'][a.model];adapter=Path(entry['path'])
    assert sha(adapter/'adapter_model.safetensors')==entry['adapter_sha256'] and sha(adapter/'adapter_config.json')==entry['config_sha256']
    source=Path(final_protocol['test_source']['path']);assert sha(source)==final_protocol['test_source']['sha256']==lock['test_source_sha256']
    rows=json.loads(source.read_text());assert len(rows)==len({r['id'] for r in rows})==len({r['group_id'] for r in rows})==71
    base_protocol=json.loads((ROOT/'configs/evaluation-protocol-v1.json').read_text());assert sha(ROOT/'configs/evaluation-protocol-v1.json')==final_protocol['base_protocol_sha256']
    for path,digest in base_protocol['code_hashes'].items():assert sha(path)==digest,path
    identity={'model':a.model,'adapter_sha256':entry['adapter_sha256'],'config_sha256':entry['config_sha256'],'format':entry['format'],
              'model_lock_sha256':sha(lockfile),'final_protocol_sha256':sha(protocolfile),'source_sha256':sha(source),'code_hashes':lock['code_hashes'],
              'inputs':'Allowlisted original image and original system/user request; JSON format suffix only for locked candidates; no labels/reference/assistant targets.'}
    identity_sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    out=OUT/'final-evaluation'/a.model
    if not a.resume:out.mkdir(parents=True,exist_ok=False)
    else:assert out.is_dir()
    worker=(out/'worker.lock').open('w');fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.resume:assert json.loads((out/'run.json').read_text())['identity']==identity
    else:
        write(out/'run.json',{'identity':identity,'started_at':stamp()});shutil.copyfile(__file__,out/'source.py')
    records=[];predictions=out/'predictions.jsonl'
    if predictions.exists():
        records=[json.loads(l) for l in predictions.open()]
        assert [r['id'] for r in records]==[r['id'] for r in rows[:len(records)]]
        assert all(r['model']==a.model and math.isfinite(r['scoring']['edited_metrics']['pixel_L1']) for r in records)
        for r in records:assert sha(r['scoring']['edited_image_path'])==r['edited_image_sha256']
    if len(records)==71:
        summary=summary_of(records,entry,source,lockfile,a.model,predictions);write(out/'summary.json',summary)
        previous=json.loads((out/'status.json').read_text());previous.update(status='completed',completed=71,summary_finalized_without_regeneration=True);write(out/'status.json',previous)
        print('All71 records complete; retained every failure, no response regenerated');return
    attempt={'pid':os.getpid(),'gpu':a.gpu,'started_at':stamp(),'resume':a.resume,'existing_rows':len(records)}
    with (out/'attempts.jsonl').open('a') as f:f.write(json.dumps(attempt)+'\n')
    status={'status':'loading','pid':os.getpid(),'gpu':a.gpu,'completed':len(records),'started_at':attempt['started_at']};write(out/'status.json',status)
    monitor=subprocess.Popen(['nvidia-smi','-i',a.gpu,'--query-gpu=timestamp,memory.used,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits','--loop=2'],stdout=(out/f'gpu-attempt-{attempt["pid"]}.csv').open('w'))
    started=time.monotonic()
    try:
        torch.set_num_threads(4);scorer=Scorer(ROOT,base_protocol);p=processor();model=load_model(adapter,trainable=True);peft_module_casting_to_bf16(model)
        for weight in model.parameters():weight.requires_grad_(False)
        model.eval()
        opened=OUT/'test-opened.json'
        with (OUT/'test-open.lock').open('a') as opening_lock:
            fcntl.flock(opening_lock,fcntl.LOCK_EX)
            if opened.exists():assert json.loads(opened.read_text())['model_lock_sha256']==sha(lockfile)
            else:write(opened,{'opened_at':stamp(),'model_lock_sha256':sha(lockfile),'protocol_sha256':sha(protocolfile),'test_sha256':sha(source),'policy':'Models and recipe now fixed; no further test-driven selection/training. If unsuccessful, this becomes regression data.'})
        for row in rows[len(records):]:
            if time.monotonic()-started>a.max_gpu_seconds:status['status']='budget_stopped';break
            key=hashlib.sha256(row['id'].encode()).hexdigest()[:20];cached=out/'responses'/f'{key}.json';cached.parent.mkdir(exist_ok=True)
            input_sha=sha(row['images'][0]);reference_sha=sha(row['extra_info']['reference_image_path'])
            if cached.exists():
                rec=json.loads(cached.read_text());assert rec['identity_sha256']==identity_sha and rec['input_sha256']==input_sha and rec['reference_sha256']==reference_sha
            else:
                set_seed(sample_seed(row['id'],20261101));begin=time.monotonic()
                rec={'id':row['id'],'group_id':row['group_id'],'model':a.model,'identity_sha256':identity_sha,'instruction':row['extra_info']['instruction'],'instruction_type':row['extra_info']['inst_type'],
                     'input_image':row['images'][0],'reference_image':row['extra_info']['reference_image_path'],'input_sha256':input_sha,'reference_sha256':reference_sha,'inference_error':None}
                try:
                    inputs={k:v.to('cuda') for k,v in encoded(p,clean(row,entry['format'])).items()}
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        generated=model.generate(**inputs,do_sample=False,max_new_tokens=512,repetition_penalty=1.,pad_token_id=p.tokenizer.pad_token_id,eos_token_id=p.tokenizer.eos_token_id,use_cache=True,disable_compile=True)
                    tokens=generated[0,inputs['input_ids'].shape[1]:].tolist();response=p.decode(tokens,skip_special_tokens=True)
                    rec.update(response_text=response,response_length=len(tokens),finish_reason='length' if len(tokens)>=512 and p.tokenizer.eos_token_id not in tokens else 'stop',length_limit_reached=len(tokens)>=512)
                except Exception as exc:rec.update(response_text='',response_length=0,finish_reason='error',inference_error=f'{type(exc).__name__}:{exc}',traceback=traceback.format_exc())
                rec['generation_seconds']=time.monotonic()-begin;write(cached,rec)
            parsed=parse_parameters(rec['response_text'],base_protocol);rec['parsing']=parsed
            rec['scoring']=scorer.score(row,parsed,out/'edited'/f'{key}.png')
            if rec['scoring'].get('metric_error'):raise RuntimeError(rec['scoring']['metric_error'])
            rec.update(empty_parameters=parsed.get('parameters')=={},all_zero_parameters=parsed.get('parameters') is not None and all(v==0 for v in parsed['parameters'].values()),edited_image_sha256=sha(rec['scoring']['edited_image_path']),completed_at=stamp())
            with predictions.open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
            records.append(rec);status.update(status='running',completed=len(records),updated_at=stamp(),elapsed_seconds=time.monotonic()-started);write(out/'status.json',status)
            if rec['inference_error']:raise RuntimeError('Generation failed; failure retained with fallback. Inspect before resuming subsequent rows.')
        if len(records)==71:
            summary=summary_of(records,entry,source,lockfile,a.model,predictions)
            assert math.isfinite(summary['mean_l1']);write(out/'summary.json',summary);status['status']='completed';print(json.dumps(summary))
    except BaseException as exc:status.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:status.update(updated_at=stamp(),elapsed_seconds=time.monotonic()-started,completed=len(records));write(out/'status.json',status);monitor.terminate();monitor.wait(timeout=10)
if __name__=='__main__':main()

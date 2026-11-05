"""Balanced opposite-instruction check on eight training images, without reference images."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import cv2
import numpy as np
import torch
from transformers import set_seed
from trl.models.utils import peft_module_casting_to_bf16
from stage2_runtime import ROOT,load_model,processor,encoded
from goal_v2_json_runtime import prepare_row,TOOLS
from goal_v2_analyze_validation import features,THRESHOLD
from evaluation_core import parse_parameters,sample_seed
OUT=ROOT/'outputs/goal-v2-20261103';DATA=ROOT/'data/goal-v2-20261103'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def prepare():
    source=DATA/'train.json';manifest=json.loads((DATA/'manifest.json').read_text());assert sha(source)==manifest['files']['train']['sha256']
    original=json.loads(source.read_text());groups=sorted({r['group_id'] for r in original},key=lambda g:hashlib.sha256(('20261103:'+g).encode()).hexdigest())[:8]
    base=[min((r for r in original if r['group_id']==g),key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest()) for g in groups]
    requests=[('brightness',1,'Make the whole image brighter.'),('brightness',-1,'Make the whole image darker.'),
              ('warmth',1,'Make the whole image warmer.'),('warmth',-1,'Make the whole image cooler.'),
              ('saturation',1,'Increase the saturation of the whole image.'),('saturation',-1,'Decrease the saturation of the whole image.')]
    rows=[]
    for row in base:
        system=next(m['content'] for m in row['conversations'] if m['role']=='system')
        for axis,sign,instruction in requests:
            user='<image> You can follow user instructions to determine the adjustments:\n'+instruction+'\nFirst, provide your thoughts about how to optimize the input image. Then, output the parameters in a JSON object format.'
            rows.append({'id':f'{row["id"]}:challenge:{axis}:{sign}','group_id':row['group_id'],'images':row['images'],
                         'conversations':[{'role':'system','content':system},{'role':'user','content':user}],
                         'extra_info':{'origin_image_path':row['images'][0],'instruction':instruction,'axis':axis,'sign':sign,'base_id':row['id']}})
    path=DATA/'instruction-challenge-v1.json';assert not path.exists();write(path,rows)
    report={'source_sha256':sha(source),'challenge_sha256':sha(path),'training_groups':8,'requests':48,'selection':'hash of source group then row ID; no image or model score selection','reference_images_or_labels_present':False,'threshold':THRESHOLD,'purpose':'Development instruction-conditioning challenge; eight source groups, not48 independent samples; not final-test evidence'}
    write(DATA/'instruction-challenge-manifest.json',report);print(json.dumps(report,indent=2))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--adapter');ap.add_argument('--name');ap.add_argument('--gpu',choices=['0','2','3']);ap.add_argument('--format',choices=['original','json']);a=ap.parse_args()
    if a.prepare:prepare();return
    assert all([a.adapter,a.name,a.gpu,a.format]) and os.environ['CUDA_VISIBLE_DEVICES']==a.gpu
    assert int(subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))<512
    source=DATA/'instruction-challenge-v1.json';manifest=json.loads((DATA/'instruction-challenge-manifest.json').read_text());assert sha(source)==manifest['challenge_sha256']
    rows=json.loads(source.read_text());assert len(rows)==48 and len({r['group_id'] for r in rows})==8
    assert all('label' not in r['extra_info'] and 'reference_image_path' not in r['extra_info'] for r in rows)
    protocol=json.loads((ROOT/'configs/evaluation-protocol-v1.json').read_text())
    for path,digest in protocol['code_hashes'].items():assert sha(path)==digest
    sys.path.insert(0,str(ROOT/'repo'));from src.image_editor import ImageEditor;from src.utils import load_and_resize_image
    out=OUT/'instruction-challenge'/a.name;out.mkdir(parents=True,exist_ok=False);shutil.copyfile(__file__,out/'source.py')
    write(out/'identity.json',{'adapter_sha256':sha(Path(a.adapter)/'adapter_model.safetensors'),'challenge_sha256':sha(source),'format':a.format,'code_sha256':sha(__file__),'reference_images_used':False})
    status={'status':'loading','pid':os.getpid(),'gpu':a.gpu,'completed':0,'started_at':stamp()};start=time.monotonic();write(out/'status.json',status)
    monitor=subprocess.Popen(['nvidia-smi','-i',a.gpu,'--query-gpu=timestamp,memory.used,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits','--loop=5'],stdout=(out/'gpu.csv').open('w'))
    records=[]
    try:
        cv2.setNumThreads(2);torch.set_num_threads(4);p=processor();model=load_model(Path(a.adapter),trainable=True);peft_module_casting_to_bf16(model)
        for weight in model.parameters():weight.requires_grad_(False)
        model.eval()
        for row in rows:
            if time.monotonic()-start>3600:raise TimeoutError('One-hour challenge cap; partial outputs preserved')
            inference=prepare_row(row,training=False) if a.format=='json' else row;seed=sample_seed(row['id'],20261103);set_seed(seed)
            inputs={k:v.to('cuda') for k,v in encoded(p,inference).items()}
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                generated=model.generate(**inputs,do_sample=False,max_new_tokens=512,repetition_penalty=1.,pad_token_id=p.tokenizer.pad_token_id,eos_token_id=p.tokenizer.eos_token_id,use_cache=True,disable_compile=True)
            tokens=generated[0,inputs['input_ids'].shape[1]:];response=p.decode(tokens,skip_special_tokens=True);parsed=parse_parameters(response,protocol)
            original=load_and_resize_image(row['images'][0]);edited=original;success=False;error=None
            if parsed['valid']:
                try:
                    np.random.seed(sample_seed(row['id'],protocol['seed']));edited=ImageEditor(row['images'][0]).process_image(parsed['parameters']);assert original.shape==edited.shape and np.isfinite(edited).all();success=True
                except Exception as exc:error=f'{type(exc).__name__}:{exc}';edited=original
            f0=features(original);f1=features(edited);axis=row['extra_info']['axis'];sign=row['extra_info']['sign'];delta=f1[axis]-f0[axis]
            target=out/'edited'/f'{hashlib.sha256(row["id"].encode()).hexdigest()[:20]}.png';target.parent.mkdir(exist_ok=True);assert cv2.imwrite(str(target),edited)
            record={'id':row['id'],'base_id':row['extra_info']['base_id'],'group_id':row['group_id'],'axis':axis,'sign':sign,'instruction':row['extra_info']['instruction'],'response_text':response,'response_length':len(tokens),'parsing':parsed,'editor_success':success,'editor_error':error,'feature_original':f0[axis],'feature_edited':f1[axis],'delta':delta,'direction_aligned':sign*delta>THRESHOLD,'pixel_identical':bool(np.array_equal(original,edited)),'edit_l1':float(np.abs(original.astype(np.float32)-edited.astype(np.float32)).mean()/255.),'edited_path':str(target)}
            with (out/'predictions.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            records.append(record);status.update(status='running',completed=len(records),elapsed_seconds=time.monotonic()-start);write(out/'status.json',status)
        pairs=[]
        for base in sorted({r['base_id'] for r in records}):
            for axis in ['brightness','warmth','saturation']:
                pos=next(r for r in records if r['base_id']==base and r['axis']==axis and r['sign']==1);neg=next(r for r in records if r['base_id']==base and r['axis']==axis and r['sign']==-1)
                dense=lambda r:[(r['parsing'].get('parameters') or {}).get(k,0) for k in TOOLS]
                pairs.append({'base_id':base,'axis':axis,'both_aligned':pos['direction_aligned'] and neg['direction_aligned'],'ordered':pos['feature_edited']-neg['feature_edited']>2*THRESHOLD,'identical_parameters':dense(pos)==dense(neg)})
        summary={'requests':48,'source_groups':8,'executable':sum(r['editor_success'] for r in records),'aligned_requests':sum(r['direction_aligned'] for r in records),'strict_pairs':sum(r['both_aligned'] for r in pairs),'ordered_pairs':sum(r['ordered'] for r in pairs),'identical_parameter_pairs':sum(r['identical_parameters'] for r in pairs),'pairs':pairs,'limits':'Balanced development challenge on8 training images; global proxies only, no reference target, aesthetics or local-edit assessment. Requests on same source are dependent.'}
        write(out/'summary.json',summary);status['status']='completed';print(json.dumps(summary,indent=2))
    except BaseException as exc:status.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:status['elapsed_seconds']=time.monotonic()-start;write(out/'status.json',status);monitor.terminate();monitor.wait(timeout=10)
if __name__=='__main__':main()

"""Frozen-A input features for a bounded numerical-action diagnostic; no test access."""
import argparse, datetime, hashlib, json, os, subprocess, time, traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from trl.models.utils import peft_module_casting_to_bf16
from stage2_runtime import ROOT, ADAPTER, load_model, processor, encoded
from goal_v2_json_runtime import prepare_row

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, d):
    t=p.with_suffix('.tmp'); t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n'); t.replace(p)
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def photo_features(path):
    with Image.open(path) as im:
        rgb=np.asarray(im.convert('RGB').resize((128,128)),dtype=np.float32)/255
    lum=rgb@np.array([.2126,.7152,.0722],np.float32)
    channels=[rgb[:,:,i] for i in range(3)]+[lum]
    return np.asarray([v for c in channels for v in [float(c.mean()),float(c.std()),*np.quantile(c,[.05,.25,.5,.75,.95]).tolist()]],np.float32)

def clean_input(row):
    # Construct an allowlist: neither reference paths nor target parameters enter encoding.
    r={'images':list(row['images']), 'conversations':[dict(m) for m in row['conversations'] if m['role'] in ('system','user')],
       'extra_info':{'origin_image_path':row['images'][0]}}
    return prepare_row(r,training=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',choices=['0','2','3'],required=True)
    ap.add_argument('--max-seconds',type=float,default=3600);a=ap.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.gpu
    assert int(subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))<512
    data=ROOT/'data/goal-v2-20261103';manifest=json.loads((data/'manifest.json').read_text())
    out=ROOT/'outputs/goal-v2-20261103/frozen-A-features-v1';out.mkdir(exist_ok=False)
    import shutil
    shutil.copyfile(__file__,out/'source.py')
    identity={'adapter_sha256':sha(ADAPTER/'adapter_model.safetensors'),'feature_code_sha256':sha(__file__),
              'json_runtime_sha256':sha(ROOT/'scripts/goal_v2_json_runtime.py'),
              'frozen':True,'inputs':'original image plus original system/user request and identical JSON schema; no reference/label',
              'features':['last prompt-token hidden state','mean non-image prompt-token hidden state','mean image-token hidden state','28 original photo statistics'],
              'splits':['train','validation'],'test_access':False}
    write(out/'identity.json',identity)
    status={'status':'loading','pid':os.getpid(),'gpu':a.gpu,'started_at':stamp(),'completed':0};start=time.monotonic();write(out/'status.json',status)
    monitor=subprocess.Popen(['nvidia-smi','-i',a.gpu,'--query-gpu=timestamp,memory.used,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits','--loop=5'],stdout=(out/'gpu.csv').open('w'))
    try:
        torch.set_num_threads(4);torch.manual_seed(20261103)
        model=load_model(ADAPTER,trainable=True);peft_module_casting_to_bf16(model)
        for p in model.parameters():p.requires_grad_(False)
        model.eval();p=processor();image_id=p.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        for split in ['train','validation']:
            source=data/f'{split}.json';assert sha(source)==manifest['files'][split]['sha256']
            rows=json.loads(source.read_text());features=[];ids=[];groups=[]
            for i,row in enumerate(rows):
                if time.monotonic()-start>a.max_seconds:raise TimeoutError('Feature-extraction time cap; partial cache preserved, no restart')
                inputs={k:v.to('cuda') for k,v in encoded(p,clean_input(row)).items()}
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    result=model.get_base_model().model(**inputs,use_cache=False,return_dict=True)
                    h=result.last_hidden_state[0].float();image_mask=inputs['input_ids'][0]==image_id
                    assert image_mask.any() and (~image_mask).any()
                    v=torch.cat([h[-1],h[~image_mask].mean(0),h[image_mask].mean(0)]).cpu().numpy()
                v=np.concatenate([v,photo_features(row['images'][0])]);assert np.isfinite(v).all()
                features.append(v);ids.append(row['id']);groups.append(row['group_id'])
                del result,h,inputs
                status.update(status='running',split=split,completed=i+1,total=len(rows),updated_at=stamp(),elapsed_seconds=time.monotonic()-start)
                if (i+1)%16==0 or i+1==len(rows):
                    np.savez(out/f'{split}-partial.npz',features=np.stack(features),ids=np.asarray(ids),groups=np.asarray(groups))
                    write(out/'status.json',status)
            (out/f'{split}-partial.npz').replace(out/f'{split}.npz')
            identity[split]={'rows':len(rows),'source_sha256':sha(source),'features_sha256':sha(out/f'{split}.npz'),'feature_dim':len(v)};write(out/'identity.json',identity)
        status['status']='completed'
    except BaseException as exc:
        status.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        status.update(updated_at=stamp(),elapsed_seconds=time.monotonic()-start);write(out/'status.json',status);monitor.terminate();monitor.wait(timeout=10)
if __name__=='__main__':main()

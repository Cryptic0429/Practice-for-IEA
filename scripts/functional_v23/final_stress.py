"""Fixed export-robustness probes; spend the remaining allowance on new generations."""
import argparse,json,signal,math,os,hashlib
from .common import *
from functional_v15.outcomes import execute,EDITOR_VERSION
from functional_v1.editor import load_rgb,save
DIR=EXP/'export-robustness'
class StopBudget(BaseException):pass

def prepare():
 from PIL import Image,ImageOps
 p=protocol();assert not (DIR/'protocol.json').exists();src=p['sources']['native_development'];assert sha(ROOT/src['path'])==src['sha256']
 source=read(ROOT/src['path']);known={'portrait':('native-dev-011','native-dev-023'),'landscape':('native-dev-070',)};groups={}
 for domain in ('portrait','landscape'):
  rows=[r for r in source if r['domain']==domain];picked=[next(r for r in rows if r['id']==cid) for cid in known[domain]]
  picked+=sorted((r for r in rows if r['group_id'] not in {c['group_id'] for c in picked}),key=lambda r:r['group_id'])[:5-len(picked)]
  groups[domain]=picked
 photos=[c for pair in zip(groups['portrait'],groups['landscape']) for c in pair];assert len({r['group_id'] for r in photos})==10
 specs=[('resize768_png',768,None),('resize768_jpeg75',768,75),('resize512_png',512,None),('resize512_jpeg75',512,75),('native_jpeg75',None,75)];queue=[]
 for cond,edge,q in specs:
  for c in photos:
   srcimage=ROOT/c['image'];assert sha(srcimage)==c['input_sha256']
   with Image.open(srcimage) as im:
    rgb=ImageOps.exif_transpose(im).convert('RGB');rgb=ImageOps.contain(rgb,(edge,edge),Image.Resampling.LANCZOS) if edge else rgb
    dest=DIR/'inputs'/f'{c["id"]}-{cond}{".jpg" if q else ".png"}';dest.parent.mkdir(parents=True,exist_ok=True);rgb.save(dest,**(dict(quality=q,subsampling=2) if q else {}))
   for lang in ('zh','en'):queue.append(dict(id=f'{c["id"]}-{cond}-{lang}',case_id=c['id'],domain=c['domain'],group_id=c['group_id'],condition=cond,language=lang,instruction=c['instructions'][lang],image=str(dest.relative_to(ROOT)),image_sha256=sha(dest),original_image=c['image'],original_sha256=c['input_sha256']))
 trial=read(EXP/'development-selection.json');r22=read(ROOT/'experiments/functional-v22-20261107/development-selection.json');models={'parent-R15':dict(path=str(PARENT.relative_to(ROOT)),sha256=PARENT_SHA)}
 for mid,m in trial['models'].items():models[mid]=dict(path=m['adapter'],sha256=m['adapter_sha256'])
 heads={mid:dict(path=m['controller'],sha256=m['controller_sha256']) for mid,m in r22['models'].items()}
 write(DIR/'protocol.json',dict(at=stamp(),status='fixed_before_export_probe_outputs',purpose='Check edit/refusal/tool/sign/strength sensitivity to common downsize/JPEG exports of known development photos. Fixed weights; no learning, tuning, acceptance or promotion use. A decision flip is a diagnostic, not automatically a quality error.',source=src,photo_groups=10,portrait=5,landscape=5,known_failure_sentinels=known,selection='Three named prior failure sentinels plus group-SHA order, alternating portrait/landscape; existing development only, no TRAIN, acceptance or user uploads.',export_conditions=[x[0] for x in specs],models=models,heads=heads,planned_inputs=len(queue),planned_inferences=len(queue)*(len(models)+len(heads)),planned_full_LM_generations=len(queue)*len(models),execution='One NF4 base, verified swap of exact single adapter state; R22 shares one actual new parent generation and target-free features for each input. Budget stop preserves completed rows; only complete five-model rows compared. No idle GPU padding.',queue=queue,script_sha256=sha(__file__),training_protocol_sha256=sha(EXP/'training-protocol.json')))
 print('Frozen100 new export inputs,10 known DEV photos,5 fixed models; no outputs generated')

def run(gpu):
 p=read(DIR/'protocol.json');assert sha(__file__)==p['script_sha256'];assert not (DIR/'generations.jsonl').exists();assert os.environ.get('CUDA_VISIBLE_DEVICES')==gpu
 import torch
 from peft import set_peft_model_state_dict
 from safetensors.torch import load_file
 from .runtime import load_model,predict,processor
 from functional_v1.runtime import verify_adapter
 from functional_v21.head import load_head,predict_cached
 from functional_v21.texture import features as texture
 from functional_v20.features import extract
 torch.set_num_threads(4);states={};heads={}
 for mid,m in p['models'].items():
  path=ROOT/m['path']/'adapter_model.safetensors';assert sha(path)==m['sha256'];states[mid]=load_file(str(path));assert len(states[mid])==116
 for mid,m in p['heads'].items():
  path=ROOT/m['path'];assert sha(path)==m['sha256'];heads[mid]=load_head(path)
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(StopBudget('Final aggregate budget stop')))
 count=0;lm=0;termination='complete_queue';verified=set()
 try:
  with GPUJob(gpu,'v23-final-export-robustness-inference',max_seconds=900) as b:
   model=load_model(adapter=PARENT);proc=processor()
   for r in p['queue']:
    assert sha(ROOT/r['image'])==r['image_sha256'];case=dict(image=r['image'],instructions={r['language']:r['instruction']});lang=r['language']
    for mid,state in states.items():
     b.tick();info=set_peft_model_state_dict(model,state);assert not info.unexpected_keys
     if mid not in verified:verify_adapter(model,ROOT/p['models'][mid]['path']);verified.add(mid)
     try:
      response,decision=predict(model,proc,case,lang);status='parsed';error=None
     except Exception as e:response=None;decision=None;status='failed';error=f'{type(e).__name__}: {e}'
     lm+=1;append(DIR/'generations.jsonl',dict(r,model_id=mid,response=response,decision=decision,status=status,error=error,at=stamp()));count+=1
     if mid=='parent-R15':
      for hid,(head,item) in heads.items():
       b.tick();hr,hd,hs,he=response,decision,status,error
       try:
        if decision and decision['action']=='edit':
         if hid==next(iter(heads)):
          cached=extract(model,proc,case,lang);tex=torch.from_numpy(texture(ROOT/r['image']))
         hr,hd=predict_cached(head,item,cached,tex,decision,'magnitude-quality')
       except Exception as e:hr=hd=None;hs='failed';he=f'{type(e).__name__}: {e}'
       append(DIR/'generations.jsonl',dict(r,model_id=hid,response=hr,decision=hd,status=hs,error=he,at=stamp()));count+=1
   del model;__import__('gc').collect();torch.cuda.empty_cache()
 except (StopBudget,BudgetExceeded) as e:termination=str(e)
 write(DIR/'execution-status.json',dict(at=stamp(),termination=termination,generations_saved=count,actual_LM_generations_saved=lm,verified_adapter_identities=sorted(verified),no_training=True,no_acceptance=True,planned_inputs=p['planned_inputs']))
 print('Saved fixed export probes:',count,'predictions;',lm,'fresh LMs;',termination)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);ap.add_argument('--gpu',choices=['0','2','3']);a=ap.parse_args();prepare() if a.mode=='prepare' else run(a.gpu)

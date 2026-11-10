"""Fresh new-LoRA generations; release GPU before CPU image rendering."""
import argparse,json,os
from .common import *
from functional_v15.outcomes import execute,EDITOR_VERSION,SCHEMA_VERSION
from functional_v1.editor import load_rgb,save
KEYS={'old-native':'old_native','native-development':'native_development','decision-development':'decision_development','development':'controlled_development'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--gpu',choices=['0','2','3'],required=True);ap.add_argument('--splits',nargs='+',choices=KEYS,default=['old-native','native-development','decision-development']);a=ap.parse_args();p=protocol();binding=read(EXP/'evaluation-implementation.json');assert all(sha(ROOT/n)==h for n,h in binding['code_hashes'].items());assert a.seed in p['training']['seeds']
 run=EXP/'runs'/f'seed-{a.seed}';status=read(run/'status.json');assert status['status']=='completed' and status['step']==250
 adapter=run/'final';assert sha(adapter/'adapter_model.safetensors')==status['adapter_sha256'];mid=f'lora-seed-{a.seed}-step250';queues=[]
 if 'development' in a.splits:
  gate=read(EXP/'stage1-gate.json');assert mid in gate['eligible']
 for split in a.splits:
  src=p['sources'][KEYS[split]];assert sha(ROOT/src['path'])==src['sha256'];cases=read(ROOT/src['path']);out=EXP/'evaluation'/EDITOR_VERSION/split/mid
  out.mkdir(parents=True,exist_ok=False);write(out/'identity.json',dict(model_id=mid,adapter=str(adapter.relative_to(ROOT)),adapter_sha256=status['adapter_sha256'],parent_sha256=PARENT_SHA,cases_sha256=src['sha256'],protocol_sha256=sha(EXP/'training-protocol.json'),runtime='functional_v23.runtime',editor=EDITOR_VERSION,schema=SCHEMA_VERSION,new_actual_LM_generations=True,input_policy='Current input photograph,literal one-language request,unchanged R15 tool/policy notes only. No target,caseID,category or reviews.'))
  queues.append((split,cases,out))
 import torch
 from .runtime import processor,load_model,predict
 torch.set_num_threads(4)
 with GPUJob(a.gpu,'v23-fresh-evaluation-'+str(a.seed),max_seconds=750) as budget:
  model=load_model(adapter=adapter);proc=processor()
  for split,cases,out in queues:
   for c in cases:
    assert sha(ROOT/c['image'])==c['input_sha256']
    for lang in ('zh','en'):
     budget.tick();row=dict(case_id=c['id'],group_id=c['group_id'],domain=c['domain'],family=c['family'],language=lang,instruction=c['instructions'][lang],model_id=mid,input_image=c['image'],input_sha256=c['input_sha256'],output_image=None,output_sha256=None,status='failed')
     try:response,decision=predict(model,proc,c,lang);row.update(response=response,decision=decision,status='parsed')
     except Exception as e:row['error']=f'{type(e).__name__}: {e}'
     append(out/'generations.jsonl',dict(row,at=stamp()))
  del model;__import__('gc').collect();torch.cuda.empty_cache()
 for split,cases,out in queues:
  rows=list(map(json.loads,(out/'generations.jsonl').read_text().splitlines()));assert len(rows)==2*len(cases)
  for r in rows:
   if r['status']=='parsed':
    try:
     result=execute(load_rgb(ROOT/r['input_image']),r['decision']);r.update(parameters=result['parameters'],execution_status=result['status'])
     if result['status']=='declined':r.update(status='declined',reason_code=result['reason_code']);assert result['result'] is None
     else:
      path=out/'images'/f'{r["case_id"]}-{r["language"]}.png';save(path,result['result']);r.update(status='executed',output_image=str(path.relative_to(ROOT)),output_sha256=sha(path))
    except Exception as e:r.update(status='failed',error=f'{type(e).__name__}: {e}')
   append(out/'predictions.jsonl',r)
  write(out/'status.json',dict(at=stamp(),status='completed_predictions',completed=len(rows),total=2*len(cases),actual_new_LM_generations=len(rows),GPU_released_before_CPU_editing=True,status_counts={s:sum(r['status']==s for r in rows) for s in ('executed','declined','failed')}))
 print('Fresh literal new-LoRA evaluation complete',mid,a.splits)
if __name__=='__main__':main()

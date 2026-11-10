"""Actual new LoRA, same literal request and strict editor as offline evaluation."""
import argparse,json,signal,sys,traceback,hashlib
from pathlib import Path
from .common import *
from functional_v1.catalog import resolve
from functional_v15.outcomes import system_prompt,execute,EDITOR_VERSION,SCHEMA_VERSION

def reply(v):print(json.dumps(v,ensure_ascii=False),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',choices=['0','2','3'],required=True);ap.add_argument('--model',required=True);a=ap.parse_args();signal.signal(signal.SIGTERM,lambda *_:sys.exit('Owned round23 trial stopped'))
 p=protocol();spec=resolve(a.model);assert spec['runtime_family']=='functional_v23';adapter=Path(spec['adapter']);assert sha(adapter/'adapter_model.safetensors')==spec['adapter_sha256']
 with GPUJob(a.gpu,'interactive-trial-'+a.model,max_seconds=300) as budget:
  from . import runtime
  from functional_v1.editor import load_rgb,save
  import torch
  torch.set_num_threads(4);model=runtime.load_model(adapter=adapter);proc=runtime.processor()
  identity=dict(model_id=a.model,display_name=spec['display_name'],adapter_sha256=spec['adapter_sha256'],parent_adapter_sha256=PARENT_SHA,runtime_family='functional_v23',runtime_sha256=sha(Path(runtime.__file__)),training_data_sha256=p['data_files']['cases.json'],protocol_sha256=sha(EXP/'training-protocol.json'),editor=EDITOR_VERSION,schema=SCHEMA_VERSION,prompt_sha256=hashlib.sha256(system_prompt().encode()).hexdigest(),editor_sha256=sha(ROOT/'scripts/functional_v15/editor.py'),inference_implementation='Actual fixed250-step continued R15 LoRA on literal original upload/request; same R23 offline runtime; no cached decision, CPU head, external rewrite or translation.')
  reply(dict(status='ready',identity=identity,model_id=a.model))
  for line in sys.stdin:
   x=json.loads(line)
   if x.get('command')=='quit':break
   output=None
   try:
    budget.tick();image=Path(x['image']);output=image.with_name('output.png');output.unlink(missing_ok=True)
    response,decision=runtime.predict(model,proc,dict(image=str(image.relative_to(ROOT)),instructions={'original':x['instruction']}),'original')
    result=execute(load_rgb(image),decision)
    if result['status']=='declined':assert result['result'] is None;reply(dict(status='declined',execution_status='declined',decision=decision,reason_code=result['reason_code'],parameters=None,response=response,output=None,output_sha256=None,identity=identity))
    else:
     save(output,result['result']);reply(dict(status='executed',execution_status=result['status'],decision=decision,parameters=result['parameters'],response=response,output=str(output),output_sha256=sha(output),identity=identity))
   except Exception as e:
    if output is not None:output.unlink(missing_ok=True)
    traceback.print_exc(file=sys.stderr);reply(dict(status='failed',error=f'{type(e).__name__}: {e}',identity=identity))
if __name__=='__main__':main()

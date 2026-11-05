"""Actually restore the chosen model's raw training state and run one bounded training update."""
import argparse,hashlib,json,math,os,random,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup
from trl.models.utils import peft_module_casting_to_bf16
from stage2_runtime import ROOT,load_model,processor,adapter_state,tensor_digest,assert_trainable
from goal_v2_json_runtime import prepare_row
from train_stage2_sft_control import build_batch
OUT=ROOT/'outputs/goal-v2-20261103'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',choices=['0','2','3'],required=True);a=ap.parse_args();assert os.environ['CUDA_VISIBLE_DEVICES']==a.gpu
    assert int(subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))<512
    lock=json.loads((OUT/'final-models-lock.json').read_text());entry=lock['models'][lock['best_by_validation']];selected=Path(entry['path']);assert selected.name=='swa'
    checkpoint=selected.parent/'checkpoint-512';assert checkpoint.is_relative_to(OUT)
    before_selected=sha(selected/'adapter_model.safetensors');assert before_selected==entry['adapter_sha256']
    out=OUT/'checkpoint-restore-smoke-v1';out.mkdir(exist_ok=False);state={'status':'loading','pid':os.getpid(),'gpu':a.gpu};write(out/'status.json',state);start=time.monotonic()
    try:
        torch.set_num_threads(4);p=processor();model=load_model(checkpoint,trainable=True);peft_module_casting_to_bf16(model);names=assert_trainable(model)
        parameters=[v for _,v in model.named_parameters() if v.requires_grad];assert len(parameters)==112
        saved_optimizer=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=True);groups=[];cursor=0
        for g in saved_optimizer['param_groups']:
            count=len(g['params']);groups.append({**{k:v for k,v in g.items() if k!='params'},'params':parameters[cursor:cursor+count]});cursor+=count
        assert cursor==112;optimizer=torch.optim.AdamW(groups);scheduler=get_cosine_schedule_with_warmup(optimizer,52,1024)
        optimizer.load_state_dict(saved_optimizer);scheduler.load_state_dict(torch.load(checkpoint/'scheduler.pt',map_location='cpu',weights_only=True))
        assert scheduler.last_epoch==512 and all(int(optimizer.state[v]['step'])==512 and optimizer.state[v]['exp_avg'].shape==v.shape for v in parameters)
        assert all(abs(g['lr']-lr)<1e-15 for g,lr in zip(optimizer.param_groups,scheduler.get_last_lr()))
        rng=torch.load(checkpoint/'rng_state.pth',map_location='cpu',weights_only=False)
        random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['cpu'])
        if isinstance(rng['cuda'],list):torch.cuda.set_rng_state_all(rng['cuda'])
        else:torch.cuda.set_rng_state(rng['cuda'])
        source=ROOT/'data/goal-v2-20261103/train3k-contrast-calibrated.json';recipe=json.loads((OUT/'development-training-lock-3k-v2.json').read_text());assert sha(source)==recipe['training_sha256']
        rows=json.loads(source.read_text())[:4];initial=tensor_digest(adapter_state(model));model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        for row in rows:
            batch={k:v.to('cuda') for k,v in build_batch(p,prepare_row(row,training=True)).items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):result=model(**batch,use_cache=False);loss=result.loss/4
            assert torch.isfinite(loss);loss.backward();losses.append(float(loss)*4)
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in parameters)
        grad=float(torch.nn.utils.clip_grad_norm_(parameters,1.));assert math.isfinite(grad);optimizer.step();scheduler.step();torch.cuda.synchronize()
        assert scheduler.last_epoch==513 and all(int(optimizer.state[v]['step'])==513 for v in parameters)
        assert all(torch.isfinite(v).all() for v in parameters) and tensor_digest(adapter_state(model))!=initial
        model.save_pretrained(out/'resumed-adapter');torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()},out/'continued-training-state.pt')
        assert sha(selected/'adapter_model.safetensors')==before_selected
        report={'passed':True,'raw_checkpoint':str(checkpoint),'source_adapter_sha256':sha(checkpoint/'adapter_model.safetensors'),'chosen_inference_adapter_unchanged_sha256':before_selected,'trainable_tensors':112,'restored_optimizer_step':512,'executed_optimizer_step':513,'restored_scheduler_step':512,'executed_scheduler_step':513,'rng_components':list(rng),'training_ids':[r['id'] for r in rows],'training_losses':losses,'grad_norm_before_clip':grad,'updated_adapter_sha256':sha(out/'resumed-adapter/adapter_model.safetensors'),'elapsed_seconds':time.monotonic()-start,'limits':['Executed restoration and one4-row optimizer update, not exact replay of the historical next dataloader batches.','Raw training state is restored; fixedSWA export remains separately derived from raw512/1024 and is never paired with the raw optimizer.','No independent-test image, reference, prediction or score used; frozen inference adapter unchanged.']}
        write(out/'summary.json',report);state['status']='completed';print(json.dumps({'passed':True,'optimizer_step':513,'scheduler_step':513,'frozen_adapter_unchanged':True}))
    except BaseException as e:state.update(status='failed',error=str(e),traceback=traceback.format_exc());raise
    finally:state['elapsed_seconds']=time.monotonic()-start;write(out/'status.json',state)
if __name__=='__main__':main()

"""CPU batch reuse after literal input/target checks, then freeze before training."""
import os,math
import torch
from .common import *
from .runtime import processor,build_batch
from functional_v15.outcomes import input_row

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (EXP/'training-protocol.json').exists()
 prior=read(ROOT/'experiments/functional-v15-20261106/batch-cache.json');assert sha(ROOT/prior['cache_path'])==prior['cache_sha256']
 oldrows=read(ROOT/'data/training/functional-v15-20261106/train-accepted.json');oldcases={c['id']:c for c in read(ROOT/'data/training/functional-v15-20261106/cases.json')}
 old=torch.load(ROOT/prior['cache_path'],map_location='cpu',weights_only=True)
 index={(r['case_id'],r['language']):(r,b) for r,b in zip(oldrows,old)}
 cases={c['id']:c for c in read(DATA/'cases.json')};rows=read(DATA/'train-accepted.json');proc=processor();cache=[];audit=[];reused=0
 for row in rows:
  c=cases[row['case_id']];lang=row['language'];assert sha(ROOT/c['image'])==c['input_sha256']
  key=(c['id'],lang);prior_row,batch=index.get(key,(None,None))
  same=prior_row is not None and prior_row['decision']==row['decision'] and input_row(oldcases[c['id']],lang)==input_row(c,lang)
  if not same:batch=build_batch(proc,c,lang,row['decision'])
  else:reused+=1
  assert batch['labels'].ne(-100).sum()>4 and batch['input_ids'].numel()<=2048
  assert all(v.device.type=='cpu' for v in batch.values() if isinstance(v,torch.Tensor))
  cache.append(batch);audit.append(dict(id=row['id'],case_id=c['id'],language=lang,reused=bool(same),assistant_tokens=int(batch['labels'].ne(-100).sum())))
 assert len(cache)==1958 and len(cases)==979
 EXP.mkdir(parents=True,exist_ok=True);torch.save(cache,EXP/'batches.pt')
 write(EXP/'batch-audit.json',dict(at=stamp(),rows=len(rows),inherited=reused,rebuilt=len(rows)-reused,records=audit,CPU_only=True))
 p22=read(ROOT/'experiments/functional-v22-20261107/training-protocol.json');used=GPUJob.consumed(read(ROOT/'experiments/functional-v1-20261104/resources.json'))
 assert used<23.4
 code=dict(p22['code_hashes'])
 for path in (ROOT/'scripts/functional_v23').glob('*.py'):code[str(path.relative_to(ROOT))]=sha(path)
 p=dict(at=stamp(),status='frozen_before_direct_QLoRA_training',hypothesis='On the exact admitted979-case pool, update the actual R15 LoRA so it can alter semantic quality-refusal actions, unlike the R22 frozen-parent magnitude head. Budget-bounded pilot, not a novel algorithm or matched-compute causal proof.',
  parent=str(PARENT.relative_to(ROOT)),parent_sha256=PARENT_SHA,data=str(DATA.relative_to(ROOT)),data_files={n:sha(DATA/n) for n in ('cases.json','train-accepted.json','admission.json')},data_counts=read(DATA/'admission.json')['counts'],
  cache_path=str((EXP/'batches.pt').relative_to(ROOT)),cache_sha256=sha(EXP/'batches.pt'),batch_audit_sha256=sha(EXP/'batch-audit.json'),
  sources=p22['sources'],sealed_acceptance_sources=p22['sealed_acceptance_sources'],
  training=dict(seeds=[20261207,20261208],max_steps=250,effective_batch=4,microbatch=1,gradient_accumulation=4,learning_rate=.0001,warmup_ratio=.05,lr_scheduler='cosine',weight_decay=0.,rank=8,alpha=16,dropout=0.,quantization='NF4 double quant, bf16 compute',trainable='112 language q/v LoRA tensors +4 visual merger LoRA tensors, one continued adapter',save='Only final fixed250 steps; optimizer/RNG included. No development-selected intermediate checkpoint.',sampling='Seeded Trainer shuffle of all1958 eligible language records. 1000 presentations per seed are a partial epoch. Actual row IDs and unique case/group/new-data coverage recorded; do not call all1958 records consumed per seed.',supervision='Only assistant outcome JSON tokens; no image target/loss, texture/risk auxiliary head, reference or review in model input.',max_seconds_per_seed=900),
  evaluation=dict(old_min=4,native_min=16,controls=20,controlled_min=81,major_allowed=0,both_seeds_required=True,order='Both fixed250 step seeds: old6/native20/control20; controlled100 only after a seed passes stage1; acceptance only after both pass development and model lock. No checkpoint/threshold edits.',judgment='Direct anonymous AI viewing, native-pixel followups as needed, exact historical disputes uncertain. No L1 selection or human/user approval.'),
  resources=dict(total_cap_GPU_hours=24,used_at_freeze=used,allowed_GPU=[0,2,3],GPU1_preserved=True,maximum_combined_training_hours=.5,trial_and_evaluation_reserve_hours=.22),code_hashes=code)
 write(EXP/'training-protocol.json',p);print('Frozen two250-step QLoRA pilots; CPU cache',reused,'inherited',len(rows)-reused,'rebuilt; used',used)
 assert not torch.cuda.is_initialized()
if __name__=='__main__':main()

"""CPU-only verification; actual data coverage is separate from eligible pool size."""
import json,hashlib
import torch
from safetensors.torch import load_file
from .common import *
def main():
 p=protocol();records=[]
 for seed in p['training']['seeds']:
  f=EXP/'runs'/f'seed-{seed}';s=read(f/'status.json');assert s['status']=='completed' and s['step']==250 and s['changed_tensors']==116 and s['language_presentations']==1000
  weight=f/'final/adapter_model.safetensors';assert sha(weight)==s['adapter_sha256'];state=load_file(str(weight));assert len(state)==116 and all(torch.isfinite(v).all() for v in state.values())
  assert sum('visual' in n for n in state)==4
  init=read(f/'initialization.json');assert init['parent_sha256']==PARENT_SHA and init['parent_tensors_copied_exactly']==116 and init['adapter_count']==1
  grad=read(f/'first-gradient-check.json');assert grad['tensors']==116 and grad['finite'] and grad['only_adapter_optimizer']
  checkpoint=f/'checkpoint-250';ts=read(checkpoint/'trainer_state.json');assert ts['global_step']==250
  optimizer=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=True);assert len(optimizer['state'])==116
  steps={int(v['step']) for v in optimizer['state'].values()};assert steps=={250}
  for name in ('optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json'):assert (checkpoint/name).is_file()
  actual=list(map(json.loads,(f/'presentations.jsonl').read_text().splitlines()));assert len(actual)==1000 and len({r['row_index'] for r in actual})==1000
  cov=read(f/'actual-data-coverage.json');assert cov['actual_language_presentations']==1000 and cov['row_sequence_sha256']==sha(f/'presentations.jsonl')
  records.append(dict(seed=seed,adapter_sha256=sha(weight),trainable_tensors=116,finite_tensors=116,changed_tensors=116,optimizer_step=250,optimizer_state_tensors=116,actual_data=cov,checkpoint_files={n:sha(checkpoint/n) for n in ('optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json')}))
 assert not torch.cuda.is_initialized();write(EXP/'training-verification.json',dict(at=stamp(),status='two_actual_fixed250_step_QLoRA_runs_verified',records=records,quality_score=False,eligible_pool_cases=979,eligible_language_rows=1958,independent_user_feedback=0))
 print('Verified two real250-step QLoRA runs,116 changed finite tensors each, optimizer/RNG and actual1000 rows per seed')
if __name__=='__main__':main()

"""Expose fixed weights for review, without declaring a functional improvement."""
import shutil
from .common import *
def main():
 p=protocol();assert read(EXP/'training-verification.json')['status']=='two_actual_fixed250_step_QLoRA_runs_verified';models={};catalog=read(ROOT/'models/catalog.json')
 assert not (EXP/'development-selection.json').exists()
 for seed in p['training']['seeds']:
  run=EXP/'runs'/f'seed-{seed}';s=read(run/'status.json');dest=ROOT/'models/from-functional-v15/sft-quality-coverage'/f'seed-{seed}';dest.mkdir(parents=True,exist_ok=False)
  for f in (run/'final').iterdir():
   if f.is_file():shutil.copy2(f,dest/f.name)
  assert sha(dest/'adapter_model.safetensors')==s['adapter_sha256'];mid=f'functional-v23-quality-{seed}'
  m=dict(model_id=mid,evaluation_id=f'lora-seed-{seed}-step250',display_name=f'第二十三轮 · 直接微调 · 种子{1 if seed==20261207 else 2}（未验收）',trial_status='实验模型、AI辅助评审；已训练固定250步，功能复核进行中，未通过验收，默认未升级。',adapter=str(dest.relative_to(ROOT)),adapter_sha256=s['adapter_sha256'],runtime_family='functional_v23',training_route='language_merger',parent=str(PARENT.relative_to(ROOT)),parent_sha256=PARENT_SHA,training_cases_sha256=p['data_files']['cases.json'],protocol_sha256=sha(EXP/'training-protocol.json'),eligible_cases=979,eligible_language_records=1958,original_groups_in_pool=240,actual_data=read(run/'actual-data-coverage.json'),seed=seed,steps=250,status='diagnostic_trial_only_review_pending',independent_user_feedback=0)
  models[mid]=m;write(dest/'metadata.json',m);catalog['models'].append(dict(id=m['adapter'],trial_id=mid,canonical_path=m['adapter'],runtime_family='functional_v23',display_name=m['display_name'],status=m['status'],metadata=m['adapter']+'/metadata.json'))
 write(ROOT/'models/catalog.json',catalog);selection=dict(at=stamp(),status='trained_fixed_weights_review_pending',qualified=False,promoted=False,selected_model_id=None,models=models,trial_default_changed=False,acceptance_status='closed_review_pending',independent_user_feedback=0,public_conclusion='第23轮两组固定250步训练完成，原始功能评测进行中；没有通过功能验收或用户认可。')
 write(EXP/'trial-registration-before-review.json',selection);write(EXP/'development-selection.json',selection);print('Two fixed diagnostic models registered; no promotion')
if __name__=='__main__':main()

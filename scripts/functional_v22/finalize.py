"""Record the fixed-seed failure without opening controlled/final acceptance."""
import shutil
from .common import *
from functional_v15.outcomes import EDITOR_VERSION

def main():
 p=protocol();ev=EXP/'evaluation'/EDITOR_VERSION;models={};results={}
 assert not (EXP/'development-selection.json').exists()
 for seed in p['training']['seeds']:
  eid=f'quality-seed-{seed}-step1160';stats={}
  for split in ('old-native','native-development','decision-development'):
   q=ev/split/eid/('decision-summary.json' if split=='decision-development' else 'direct-functional-summary.json');s=read(q)
   stats[split]=dict(completed=s.get('fully_completed_both_languages',s.get('passed_both_languages')),major=s.get('major_quality_issues',0),path=str(q.relative_to(ROOT)),sha256=sha(q))
  assert stats['native-development']['completed']<16
  results[eid]=stats
  src=EXP/'runs'/f'seed-{seed}'/'checkpoint-step-1160.pt';dest=ROOT/'models/from-functional-v15/quality-coverage-controller'/f'seed-{seed}'
  dest.mkdir(parents=True,exist_ok=False);shutil.copy2(src,dest/'controller.pt');mid=f'functional-v22-quality-{seed}'
  m=dict(model_id=mid,evaluation_id=eid,display_name=f'第二十二轮 · 质量覆盖 · 种子{1 if seed==20261205 else 2}（未通过）',
   trial_status=f'实验模型、AI辅助评审；原始开发{stats["native-development"]["completed"]}/20、旧{stats["old-native"]["completed"]}/6，重大0；未达到预设门槛，不保证可靠拒绝，默认未升级。',
   controller=str((dest/'controller.pt').relative_to(ROOT)),controller_sha256=sha(src),policy='magnitude-quality',runtime_family='functional_v22',parent=str(PARENT.relative_to(ROOT)),parent_sha256=PARENT_SHA,
   protocol_sha256=sha(EXP/'training-protocol.json'),training_cases_sha256=sha(DATA/'cases.json'),semantic_condition_cases=979,language_records=1958,original_groups=240,
   actual_updates=1160,actual_language_presentations=36652,trainable_parameters=463754,CPU_training=True,step=1160,seed=seed,
   status='diagnostic_trial_only_not_promoted',acceptance_status='closed_stage1_failed',independent_user_feedback=0)
  models[mid]=m;write(dest/'metadata.json',m)
 catalog=read(ROOT/'models/catalog.json')
 for mid,m in models.items():
  assert mid not in {a.get('trial_id',a.get('id')) for a in catalog['models']}
  catalog['models'].append(dict(id=m['controller'],trial_id=mid,canonical_path=str((ROOT/m['controller']).parent.relative_to(ROOT)),display_name=m['display_name'],runtime_family='functional_v22',status=m['status'],metadata=str((ROOT/m['controller']).parent.relative_to(ROOT))+'/metadata.json'))
 write(ROOT/'models/catalog.json',catalog)
 write(EXP/'stage1-gate.json',dict(at=stamp(),eligible=[],results=results,qualified=False,thresholds=p['evaluation']))
 write(EXP/'development-selection.json',dict(at=stamp(),status='failed_fixed_two_seed_native_gate',qualified=False,promoted=False,selected_model_id=None,models=models,results=results,trial_default_changed=False,acceptance_status='closed_stage1_failed',controlled_evaluation_started=False,independent_user_feedback=0,
  public_conclusion='第二十二轮两组固定种子原始开发15/20、13/20，旧回归4/6、6/6，重大0，控制均20/20；均未达到原始16/20门槛，停止进入受控/最终验收。质量标签覆盖扩大，但没有得到稳定可用模型。'))
 print('R22 fixed two-seed gate failed;2 diagnostic models registered, acceptance sealed')
if __name__=='__main__':main()

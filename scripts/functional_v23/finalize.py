"""Report both fixed seeds as observed; close route with no new deployment."""
from .common import *
from functional_v15.outcomes import EDITOR_VERSION

def main():
 p=protocol();s=read(EXP/'development-selection.json');assert s['status']=='trained_fixed_weights_review_pending'
 result={};eligible=[];ev=EXP/'evaluation'/EDITOR_VERSION
 for mid,m in s['models'].items():
  eid=m['evaluation_id'];stats={}
  for split in ('old-native','native-development','decision-development'):
   path=ev/split/eid/('decision-summary.json' if split=='decision-development' else 'direct-functional-summary.json');r=read(path);assert r['status']=='completed'
   stats[split]=dict(completed=r.get('fully_completed_both_languages',r.get('passed_both_languages')),major=r.get('major_quality_issues',0),path=str(path.relative_to(ROOT)),sha256=sha(path))
  old,native,control=[stats[k] for k in ('old-native','native-development','decision-development')];passed=old['completed']>=4 and native['completed']>=16 and control['completed']==20 and not(old['major'] or native['major'])
  if passed:eligible.append(eid)
  m.update(trial_status=f'实验模型、AI辅助评审；旧回归{old["completed"]}/6、原始开发{native["completed"]}/20、控制{control["completed"]}/20。两组未同时过关，夜景画质争议保留不确定。无最终验收，预算已结算；可看已保存结果。',status='diagnostic_only_budget_closed',acceptance_status='closed_paired_route_not_qualified',stage1_eligible=passed)
  write(ROOT/m['adapter']/'metadata.json',m);result[eid]=stats
 assert len(eligible)==1 and eligible==['lora-seed-20261208-step250']
 assert not (ev/'development').exists() and not (ev/'acceptance').exists() and not (ev/'native-acceptance').exists()
 write(EXP/'stage1-gate.json',dict(at=stamp(),eligible=eligible,paired_route_eligible=False,results=result,required_both_seeds=True,criteria=p['evaluation']))
 s.update(at=stamp(),status='paired_route_not_qualified_budget_closed',results=result,stage1_eligible=eligible,controlled_evaluation_started=False,acceptance_status='closed_paired_route_not_qualified',qualified=False,promoted=False,selected_model_id=None,
  public_conclusion='第23轮原始开发两组均16/20，控制均20/20，旧回归分别3/6、4/6；第二组达到单种子第一阶段门槛，第一组未达到旧回归门槛。夜景中文仍大幅提亮、英文拒绝；中文结果同像素评审分歧保留不确定，重大0并非安全保证。两组路线未同时通过，预算结算，未评受控100或开封最终验收；默认未升级。')
 write(EXP/'development-selection.json',s);catalog=read(ROOT/'models/catalog.json')
 for item in catalog['models']:
  if item.get('trial_id') in s['models']:item['status']='diagnostic_only_budget_closed'
 write(ROOT/'models/catalog.json',catalog);print(s['public_conclusion'])
if __name__=='__main__':main()

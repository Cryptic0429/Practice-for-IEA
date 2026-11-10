"""Register fixed final heads only after actual declared checkpoints fail gates."""
import shutil
from .common import ROOT,EXP,DATA,ROUTES,PARENT,PARENT_SHA,read,write,sha,stamp,protocol
from .visual_review import EVAL

def main():
    p=protocol();assert not (EXP/'development-selection.json').exists()
    old=read(EXP/'old-screen-gate.json');results={}
    for route in ROUTES:
        for epoch in (20,80):
            eid=f'{route.replace("_","-")}-e{epoch}';o=read(EVAL/'old-native'/eid/'direct-functional-summary.json')
            row=dict(old_completed=o['fully_completed_both_languages'],old_major=o['major_quality_issues'],native_summary=None)
            passed=old['results'][eid]['eligible']
            if passed:
                n=read(EVAL/'native-development'/eid/'direct-functional-summary.json');c=read(EVAL/'decision-development'/eid/'decision-summary.json')
                passed=n['fully_completed_both_languages']>=16 and n['major_quality_issues']==0 and c['passed_both_languages']==20
                row.update(native_certified_complete=n['fully_completed_both_languages'],native_major=n['major_quality_issues'],decision_controls=c['passed_both_languages'],
                    native_count_is_conservative_with_unresolved_details=True,native_summary_sha256=sha(EVAL/'native-development'/eid/'direct-functional-summary.json'))
            assert not passed,'A qualifier needs real controlled development, replication and acceptance; no failed-registration shortcut'
            row['qualified']=False;results[eid]=row
    models={};catalog=read(ROOT/'models/catalog.json')
    for route in ROUTES:
        run=EXP/'runs'/route;assert read(run/'status.json')['status']=='completed'
        source=run/'checkpoint-80.pt';dest=ROOT/'models/from-functional-v15'/f'continuous-controller-{route.replace("_","-")}'/'seed-01'
        dest.mkdir(parents=True,exist_ok=False);shutil.copy2(source,dest/'controller.pt');assert sha(source)==sha(dest/'controller.pt')
        mid=f'functional-v20-{route.replace("_","-")}-20261203';eid=route.replace('_','-')+'-e80';old_count=results[eid]['old_completed']
        title='最后位置' if route=='last' else '原始请求均值'
        status=f'实验模型，AI辅助评审；旧回归{old_count}/6。'+('原始开发确认1例重大噪声损伤，其余部分原尺寸质量仍不确定，未通过；' if route=='last' else '未通过初筛，未扩大原始开发评测；')+'固定第80轮仅诊断试用，默认未升级。'
        meta=dict(model_id=mid,evaluation_id=eid,display_name=f'第二十轮 · {title}预测头 · 第80轮（未通过）',trial_status=status,
            controller=str((dest/'controller.pt').relative_to(ROOT)),controller_sha256=sha(source),route=route,runtime_family='functional_v20',
            parent=str(PARENT.relative_to(ROOT)),parent_sha256=PARENT_SHA,parent_identity='Actual frozen R15e1; earlier-start-named directory',
            kind='frozen_VLM_plus_CPU_controller_not_LoRA_only',parameters=466088,checkpoint=str(source.relative_to(ROOT)),epoch=80,step=4640,
            seed=20261203,training_protocol_sha256=sha(EXP/'training-protocol.json'),training_cases=str((DATA/'cases.json').relative_to(ROOT)),
            training_cases_sha256=p['feature_sources']['train']['sha256'],semantic_cases=925,language_records=1850,original_groups=240,
            native_summary=results[eid],status='diagnostic_trial_only_not_promoted',
            diagnostic_choice='All4 fixed checkpoints failed predeclared gates. Both fixed final80 heads exposed; no winner or checkpoint selection.',
            inference='Same frozen R15 image/literal-request/tools encoding and current-input16RGBstats as cached eval; R20 structured head and R15 editor. No external translation/rewrite.',
            acceptance_status='closed_not_evaluated',independent_user_feedback=0)
        write(dest/'metadata.json',meta);models[mid]=meta
        assert mid not in {m.get('trial_id',m.get('id')) for m in catalog['models']}
        catalog['models'].append(dict(id=str(dest.relative_to(ROOT/'models')),trial_id=mid,canonical_path=str(dest.relative_to(ROOT)),
             display_name=meta['display_name'],kind='frozen_VLM_plus_learned_controller',parent=meta['parent'],runtime_family='functional_v20',
             status=meta['status'],controller_sha256=meta['controller_sha256'],metadata=str((dest/'metadata.json').relative_to(ROOT))))
    conclusion='第二十轮最后位置路线旧6/6，但两个检查点均确认星空提亮重大噪声损伤；原始开发原尺寸未决项仍不确定。请求均值路线旧3/6。四个检查点均未通过，未受控开发、复核或开封验收，默认未升级。'
    write(EXP/'development-selection.json',dict(at=stamp(),status='failed_practical_gate',qualified=False,promoted=False,selected_model_id=None,
        trial_default_changed=False,models=models,checkpoint_results=results,public_conclusion=conclusion,acceptance_status='closed_not_evaluated',
        controlled_evaluation_started=False,seed_replication_started=False,independent_user_feedback=0,
        native_review_scope=read(EXP/'native-detail-stop.json')))
    catalog['changed_on']='2026-11-07';write(ROOT/'models/catalog.json',catalog)
    print(conclusion)

if __name__=='__main__':main()

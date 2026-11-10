"""Publish exact diagnostic trial identities after the locked replication fails."""
import shutil
from .common import ROOT,EXP,DATA,PARENT,PARENT_SHA,protocol,read,write,sha,stamp
from .visual_review import EVAL


def main():
    p=protocol();assert not (EXP/'development-selection.json').exists()
    lock=read(EXP/'replication-lock.json');assert lock['model']=='magnitude-quality-e20' and lock['epoch']==20
    replica=EXP/'replication/evaluation'/EVAL.name
    rn=read(replica/'native-development/magnitude-quality-e20/direct-functional-summary.json')
    assert rn['status']=='completed' and rn['major_quality_issues']>0
    assert not (replica/'development').exists(),'No controlled replicate claim after failed native gate'
    for receipt in ('training-verification.json','replication-training-verification.json'):
        assert read(EXP/receipt)['status']=='actual_CPU_training_verified'
    results={}
    for epoch in (20,80):
        for policy in ('magnitude','magnitude-quality'):
            eid=f'{policy}-e{epoch}';row={}
            for split in ('old-native','native-development','decision-development'):
                path=EVAL/split/eid/('decision-summary.json' if split=='decision-development' else 'direct-functional-summary.json')
                s=read(path);row[split]=dict(completed=s.get('fully_completed_both_languages',s.get('passed_both_languages')),
                    major=s.get('major_quality_issues',0),path=str(path.relative_to(ROOT)),sha256=sha(path))
            path=EVAL/'development'/eid/'direct-functional-summary.json'
            if path.exists():
                s=read(path);row['development']=dict(completed=s['fully_completed_both_languages'],major=s['major_quality_issues'],path=str(path.relative_to(ROOT)),sha256=sha(path))
            results[eid]=row
    models={};catalog=read(ROOT/'models/catalog.json')
    for seed,policies in [(20261205,('magnitude','magnitude-quality')),(20261206,('magnitude-quality',))]:
        source=EXP/'runs'/f'seed-{seed}'/'checkpoint-20.pt'
        dest=ROOT/'models/from-functional-v15/semantic-preserving-controller'/('seed-01' if seed==20261205 else 'seed-02')
        dest.mkdir(parents=True,exist_ok=False);shutil.copy2(source,dest/'controller.pt');assert sha(source)==sha(dest/'controller.pt')
        folder_models={}
        for policy in policies:
            tag='quality' if policy=='magnitude-quality' else 'magnitude';mid=f'functional-v21-{tag}-{seed}'
            title='幅度＋质量拒绝' if tag=='quality' else '仅幅度对照'
            state=('首种子原始16/20、旧5/6、受控82/100且重大0，但换种子未重复质量拒绝，路线未通过；' if tag=='quality' and seed==20261205 else
                '原始开发确认星空提亮重大噪声损伤；')
            status='实验模型，AI辅助评审；'+state+'固定第20轮仅诊断试用，不保证可靠拒绝，默认未升级。'
            meta=dict(model_id=mid,evaluation_id=f'{policy}-e20',seed=seed,policy=policy,display_name=f'第二十一轮 · {title} · 种子{1 if seed==20261205 else 2}（未通过复核）',
                trial_status=status,controller=str((dest/'controller.pt').relative_to(ROOT)),controller_sha256=sha(source),runtime_family='functional_v21',
                parent=str(PARENT.relative_to(ROOT)),parent_sha256=PARENT_SHA,parent_identity='Actual frozen R15e1, earlier-start-named directory',
                kind='frozen_parent_LM_semantics_plus_CPU_magnitude_quality_head',trainable_parameters=463754,epoch=20,step=1160,
                protocol_sha256=sha(EXP/'training-protocol.json'),training_cases_sha256=p['sources']['train']['sha256'],
                training_cases=str((DATA/'cases.json').relative_to(ROOT)),semantic_cases=925,language_records=1850,original_groups=240,
                checkpoint=str(source.relative_to(ROOT)),policy_weight_sharing='Seed1 magnitude andquality variants use byte-identical head weights. Seed2 is an independently trained locked replication.',
                evaluation_namespace=str((EXP if seed==20261205 else EXP/'replication').relative_to(ROOT)),
                inference='Actual frozen R15 LM on literal original upload/request/tools, then same R21 magnitude/risk head and R15 editor. Parent keep/declines preserved; no external rewrite or translation.',
                acceptance_status='closed_replication_failed',status='diagnostic_trial_only_not_promoted',independent_user_feedback=0)
            models[mid]=meta;folder_models[mid]=meta
            assert mid not in {m.get('trial_id',m.get('id')) for m in catalog['models']}
            catalog['models'].append(dict(id=str(dest.relative_to(ROOT/'models'))+'/'+tag,trial_id=mid,canonical_path=str(dest.relative_to(ROOT)),
                display_name=meta['display_name'],kind=meta['kind'],parent=meta['parent'],runtime_family='functional_v21',status=meta['status'],
                controller_sha256=meta['controller_sha256'],metadata=str((dest/'metadata.json').relative_to(ROOT))))
        write(dest/'metadata.json',dict(models=folder_models,shared_controller_sha256=sha(source)))
    conclusion='第二十一轮首种子质量策略原始16/20、旧5/6、受控82/100且重大0，但锁定第20轮后换种子复核再次出现重大夜景噪声，质量拒绝不稳定，路线未通过。未开封最终验收，默认未升级。'
    write(EXP/'development-selection.json',dict(at=stamp(),status='failed_locked_seed_replication',qualified=False,promoted=False,selected_model_id=None,
        provisional_development_candidate='functional-v21-quality-20261205',trial_default_changed=False,models=models,checkpoint_results=results,
        replication_native=dict(completed_lower_bound=rn['fully_completed_both_languages'],major=rn['major_quality_issues'],
            summary_sha256=sha(replica/'native-development/magnitude-quality-e20/direct-functional-summary.json'),
            unresolved_detail_scope='Other native details stay uncertain after confirmed major failure; lower bound is not an equal-precision comparison with fully resolved initial16/20.'),
        public_conclusion=conclusion,acceptance_status='closed_replication_failed',independent_user_feedback=0,
        controlled_evaluation_started=True,seed_replication_started=True,seed_replication_passed=False,replicate_controlled_started=False))
    catalog['changed_on']='2026-11-07';write(ROOT/'models/catalog.json',catalog);print(conclusion)


if __name__=='__main__':main()

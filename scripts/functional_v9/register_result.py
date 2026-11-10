"""Register completed diagnostic endpoints only after all frozen functional reviews."""
import shutil
from functional_v1.common import ROOT,read,write,sha,stamp,GPUJob,EXP as BUDGET_EXP
from functional_v7.outcomes import EDITOR_VERSION,SCHEMA_VERSION
from .runtime import EXP,DATA

EVAL=EXP/'evaluation'/EDITOR_VERSION


def main():
    destination=EXP/'development-selection.json';assert not destination.exists()
    assert read(EXP/'checkpoint-queue.json')['status']=='completed'
    protocol=read(EXP/'training-protocol.json')
    for name,digest in protocol['code_hashes'].items():assert sha(ROOT/name)==digest
    models=['base-init','author-init']+[f'{route}-e{epoch}' for epoch in range(1,5) for route in ('base','author')]
    results={}
    summaries={}
    for model in models:
        native=read(EVAL/'native-development'/model/'direct-functional-summary.json')
        old=read(EVAL/'old-native'/model/'direct-functional-summary.json')
        decision=read(EVAL/'decision-development'/model/'decision-summary.json')
        assert native['status']==old['status']=='completed' and not native['missing_unique_images'] and not old['missing_unique_images']
        record=dict(native_completed=native['fully_completed_both_languages'],native_major=native['major_quality_issues'],
            old_completed=old['fully_completed_both_languages'],old_major=old['major_quality_issues'],
            decision_controls=decision['passed_both_languages'],feasible_native_refusals=native['declined_semantic_cases'],
            feasible_old_refusals=old['declined_semantic_cases'])
        record['stage1_passed']=record['native_completed']>=16 and record['old_completed']>=4 and record['native_major']==record['old_major']==0 and record['decision_controls']==20
        results[model]=record;summaries[model]=dict(native=native,old=old,decisions=decision)
    passed=[name for name,r in results.items() if '-e' in name and r['stage1_passed']]
    write(EXP/'stage1-decision.json',dict(at=stamp(),results=results,qualified_checkpoints=passed,
        evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),training_protocol_sha256=sha(EXP/'training-protocol.json'),
        stage2_required_before_selection=True,L1_used=False,independent_user_feedback=0))
    if passed:
        print('Stage1 candidates require frozen stage2, replication and lock before acceptance:',passed)
        return
    diagnostic={};admission=read(DATA/'admission.json')
    for origin in ('base','author'):
        run=EXP/'runs'/f'{origin}-seed-20261115';status=read(run/'status.json')
        assert status['status']=='completed' and status['step']==924 and status['changed_language_tensors']==112
        checkpoint=run/'checkpoint-924'
        parent='from-base' if origin=='base' else 'from-author-iea'
        target=ROOT/'models'/parent/'sft-fresh-outcome/seed-01'
        assert not target.exists();target.mkdir(parents=True)
        for name in ('adapter_model.safetensors','adapter_config.json'):
            shutil.copy2(checkpoint/name,target/name)
            assert sha(target/name)==sha(checkpoint/name)
        digest=sha(target/'adapter_model.safetensors');assert digest==status['adapter_sha256']
        model_id=f'functional-v9-{origin}-20261115';evaluation_id=f'{origin}-e4'
        meta=dict(model_id=model_id,evaluation_id=evaluation_id,
            display_name='第九轮 · '+('原始Qwen起点' if origin=='base' else '作者IEA起点')+' · 第4轮（未通过）',
            trial_status='实验模型，AI辅助评审；未过功能门槛，仍可能压暗人脸、放大夜景噪声或漏掉要求。拒绝判定不保证识别所有画质风险。'+('作者完整权重存在历史GIER原图重叠。' if origin=='author' else ''),
            adapter=str(target.relative_to(ROOT)),adapter_sha256=digest,
            full_weight_origin=protocol['origins'][origin],origin=origin,parent=protocol['origins'][origin]['path'],
            initial_adapter='fresh new language q/v LoRA; no parent adapter transferred or stacked',
            checkpoint=str(checkpoint.relative_to(ROOT)),epoch=4,step=924,runtime_family='functional_v9',training_route=origin,
            editor=EDITOR_VERSION,schema=SCHEMA_VERSION,dataset_sha256=protocol['dataset_sha256'],cases_sha256=protocol['cases_sha256'],
            actual_training_data=str((DATA/'train-accepted.json').relative_to(ROOT)),semantic_cases=462,training_records=924,
            original_photo_groups=160,photo_domains=dict(portrait=80,landscape=80),outcome_counts=admission['outcome_counts'],
            original_GIER_English_cases=97,AI_authored_cases=365,status='diagnostic_trial_only_not_promoted',
            diagnostic_choice='Both final epoch4 endpoints are exposed to inspect training outcome; neither is a qualified checkpoint winner.',
            load_method='Load the EXACT audited '+origin+' full weights, then this ONE adapter. Author-origin adapter MUST NOT be attached to original-Qwen weights.',
            native_development_summary=summaries[evaluation_id]['native'],old_native_summary=summaries[evaluation_id]['old'],
            decision_summary=summaries[evaluation_id]['decisions'],training_protocol_sha256=sha(EXP/'training-protocol.json'),
            evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),origin_audit_sha256=sha(EXP/'origin-audit.json'))
        write(target/'metadata.json',meta);diagnostic[model_id]=meta
    resource=read(BUDGET_EXP/'resources.json')
    write(destination,dict(at=stamp(),status='failed_practical_gate',qualified=False,promoted=False,selected_model_id=None,
        trial_default_changed=False,stage1_qualified_checkpoints=[],results=results,
        controlled_development_status='not_run_all_candidates_failed_predeclared_stage1',seed_replication_started=False,
        acceptance_status='closed_not_evaluated',models=diagnostic,
        public_conclusion='本轮各检查点均未通过原始请求与零重大副作用门槛，未升级默认模型。下列两个第4轮权重仅供训练终点诊断试用，不是合格冠军；验收未开封。',
        explanation='Better schema compliance is not equivalent to completing photographic edits. No L1 or training loss selection; feasible-task refusals stay uncompleted. Author source overlap disclosed.',
        training_protocol_sha256=sha(EXP/'training-protocol.json'),evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),
        all_checkpoint_results_visible=True,independent_user_feedback=0,resource_snapshot=dict(occupied_GPU_hours=GPUJob.consumed(resource),cap=16)))
    catalog_path=ROOT/'models/catalog.json';catalog=read(catalog_path)
    existing={entry.get('trial_id',entry.get('id')) for entry in catalog['models']}
    for model_id,meta in diagnostic.items():
        assert model_id not in existing
        catalog['models'].append(dict(id=str((ROOT/meta['adapter']).relative_to(ROOT/'models')),trial_id=model_id,
            canonical_path=meta['adapter'],display_name=meta['display_name'],kind='lora_adapter',
            parent=meta['parent'],runtime_family='functional_v9',origin=meta['origin'],
            status=meta['status'],metadata=meta['adapter']+'/metadata.json',adapter_sha256=meta['adapter_sha256']))
    catalog['changed_on']='2026-11-06';write(catalog_path,catalog)
    print('All stage1 checkpoints failed; registered two origin-specific epoch4 diagnostic models, default unchanged.')

if __name__=='__main__':main()

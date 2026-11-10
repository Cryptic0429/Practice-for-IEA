"""Register two fixed endpoints only after all prescribed stage1 image reviews."""
import shutil
from functional_v1.common import ROOT,read,write,sha,stamp,GPUJob,EXP as BUDGET_EXP
from functional_v7.outcomes import EDITOR_VERSION,SCHEMA_VERSION
from .runtime import EXP,DATA
EVAL=EXP/'evaluation'/EDITOR_VERSION


def main():
    destination=EXP/'development-selection.json';assert not destination.exists()
    assert read(EXP/'final-queue.json')['status']=='completed'
    protocol=read(EXP/'training-protocol.json')
    for name,digest in protocol['code_hashes'].items():assert sha(ROOT/name)==digest
    results={};summaries={}
    for route in ('language','language_merger'):
        model=route.replace('_','-')+'-e2'
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
    passed=[m for m,r in results.items() if r['stage1_passed']]
    write(EXP/'stage1-decision.json',dict(at=stamp(),results=results,qualified_checkpoints=passed,
        evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),training_protocol_sha256=sha(EXP/'training-protocol.json'),
        stage2_required_before_selection=True,L1_used=False,independent_user_feedback=0))
    if passed:
        print('Stage1 passed; frozen stage2, seed replication and lock still required:',passed);return
    diagnostic={};admission=read(DATA/'admission.json')
    for route in ('language','language_merger'):
        run=EXP/'runs'/f'{route}-seed-20261121';status=read(run/'status.json')
        assert status['status']=='completed' and status['step']==240 and status['changed_language_tensors']==112
        assert status['changed_visual_merger_tensors']==(4 if route=='language_merger' else 0)
        checkpoint=run/'checkpoint-240'
        target=ROOT/'models/from-functional-v9'/('sft-condition-'+route.replace('_','-'))/'seed-01'
        assert not target.exists();target.mkdir(parents=True)
        for name in ('adapter_model.safetensors','adapter_config.json'):
            shutil.copy2(checkpoint/name,target/name);assert sha(target/name)==sha(checkpoint/name)
        digest=sha(target/'adapter_model.safetensors');assert digest==status['adapter_sha256']
        evaluation_id=route.replace('_','-')+'-e2';model_id='functional-v10-'+route.replace('_','-')+'-20261121'
        meta=dict(model_id=model_id,evaluation_id=evaluation_id,
            display_name='第十轮 · '+('语言层＋视觉连接层' if route=='language_merger' else '语言层')+' · 第2轮（未通过）',
            trial_status='实验模型，AI辅助评审；未过功能门槛，仍可能漏项、损伤画质或对可完成任务误拒绝。合成噪声训练不保证未知照片上的画质拒绝可靠。',
            adapter=str(target.relative_to(ROOT)),adapter_sha256=digest,origin='base',full_weight_origin=protocol['origin_identity'],
            parent=protocol['parent_adapter'],parent_sha256=protocol['parent_sha256'],parent_status=protocol['parent_status'],
            initial_adapter='One adapter;112 language tensors copied exactly; additional visual-merger B zero; no stacking',
            checkpoint=str(checkpoint.relative_to(ROOT)),epoch=2,step=240,runtime_family='functional_v10',training_route=route,
            editor=EDITOR_VERSION,schema=SCHEMA_VERSION,dataset_sha256=protocol['dataset_sha256'],cases_sha256=protocol['cases_sha256'],
            actual_training_data=str((DATA/'train-accepted.json').relative_to(ROOT)),semantic_cases=240,training_records=480,
            original_photo_groups=60,photo_domains=dict(portrait=30,landscape=30),parent_cumulative_original_groups=160,
            new_original_groups=0,outcome_counts=admission['outcome_counts'],original_GIER_English_cases=27,AI_authored_cases=213,
            status='diagnostic_trial_only_not_promoted',diagnostic_choice='Final epoch2 checkpoint fixed before training, not chosen by loss or development ranking.',
            load_method='Original Qwen full weights plus this ONE adapter; same R7 prompt/parser/editor as offline evaluation.',
            native_development_summary=summaries[evaluation_id]['native'],old_native_summary=summaries[evaluation_id]['old'],decision_summary=summaries[evaluation_id]['decisions'],
            training_protocol_sha256=sha(EXP/'training-protocol.json'),evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),
            origin_audit_sha256=protocol['origin_audit_sha256'])
        write(target/'metadata.json',meta);diagnostic[model_id]=meta
    write(destination,dict(at=stamp(),status='failed_practical_gate',qualified=False,promoted=False,selected_model_id=None,
        trial_default_changed=False,stage1_qualified_checkpoints=[],results=results,controlled_development_status='not_run_both_final_candidates_failed_stage1',
        seed_replication_started=False,acceptance_status='closed_not_evaluated',models=diagnostic,
        public_conclusion='两条预先固定的第2轮模型均未通过原始请求与零重大副作用门槛，没有升级默认模型；仅供诊断试用，验收未开封。',
        explanation='Training completion, valid decision JSON or synthetic-condition fit cannot establish practical usefulness.',
        occupied_gpu_hours_at_registration=GPUJob.consumed(read(BUDGET_EXP/'resources.json')),independent_user_feedback=0))
    path=ROOT/'models/catalog.json';catalog=read(path)
    existing={entry.get('trial_id',entry.get('id')) for entry in catalog['models']}
    for mid,m in diagnostic.items():
        assert mid not in existing
        catalog['models'].append(dict(id=str((ROOT/m['adapter']).relative_to(ROOT/'models')),trial_id=mid,
            canonical_path=m['adapter'],display_name=m['display_name'],kind='lora_adapter',
            parent=m['parent'],runtime_family='functional_v10',origin='base',status=m['status'],
            metadata=m['adapter']+'/metadata.json',adapter_sha256=m['adapter_sha256']))
    catalog['changed_on']='2026-11-06';write(path,catalog)
    print('Both fixed final candidates failed; two diagnostic adapters registered; default unchanged.')

if __name__=='__main__':main()

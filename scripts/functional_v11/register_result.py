"""Register the fixed final adapter with actual functional results and lineage."""
import shutil
from functional_v1.common import ROOT,read,write,sha,stamp,GPUJob,EXP as BUDGET_EXP
from .runtime import EXP,DATA
from .outcomes import EDITOR_VERSION,SCHEMA_VERSION


def main():
    destination=EXP/'development-selection.json';assert not destination.exists()
    stage1=read(EXP/'stage1-decision.json')
    protocol=read(EXP/'training-protocol.json')
    for path,digest in protocol['code_hashes'].items():assert sha(ROOT/path)==digest
    main_id='language-merger-e1';r=stage1['results'][main_id]
    stage2=None
    if r['stage1_passed']:
        stage2=read(EXP/'evaluation'/EDITOR_VERSION/'development'/main_id/'direct-functional-summary.json')
        assert stage2['status']=='completed' and not stage2['missing_unique_images']
    qualified=bool(r['stage1_passed'] and stage2['fully_completed_both_languages']>=81 and stage2['major_quality_issues']==0)
    run=EXP/'runs/language_merger-seed-20261123';status=read(run/'status.json')
    assert status['status']=='completed' and status['step']==336
    assert status['changed_language_tensors']==112 and status['changed_visual_merger_tensors']==4
    checkpoint=run/'checkpoint-336'
    target=ROOT/'models/from-functional-v10/sft-negative-gain-correction/seed-01'
    assert not target.exists();target.mkdir(parents=True)
    for name in ('adapter_model.safetensors','adapter_config.json'):
        shutil.copy2(checkpoint/name,target/name);assert sha(target/name)==sha(checkpoint/name)
    digest=sha(target/'adapter_model.safetensors');assert digest==status['adapter_sha256']
    admission=read(DATA/'admission.json');mid='functional-v11-negative-gain-20261123'
    meta=dict(model_id=mid,evaluation_id=main_id,display_name='第十一轮 · 负曝光与指令纠错 · 第1轮'+('（待复核）' if qualified else '（未通过）'),
        trial_status='实验模型，AI辅助评审；'+('开发门槛通过，尚待换种子复核与冻结验收。' if qualified else '未通过功能门槛，仅供诊断试用。')+'未知照片仍可能漏项、损伤画质或误拒绝；拒绝时不返回结果图。',
        adapter=str(target.relative_to(ROOT)),adapter_sha256=digest,origin='base',full_weight_origin=protocol['origin_identity'],
        parent=protocol['parent_adapter'],parent_sha256=protocol['parent_sha256'],parent_status=protocol['parent_status'],
        initial_adapter='One adapter; all116 trained parent tensors copied exactly, including4 visual merger tensors; no stacking or new zero-B visual modules.',
        checkpoint=str(checkpoint.relative_to(ROOT)),epoch=1,step=336,runtime_family='functional_v11',training_route='language_merger',
        editor=EDITOR_VERSION,schema=SCHEMA_VERSION,dataset_sha256=protocol['dataset_sha256'],cases_sha256=protocol['cases_sha256'],
        actual_training_data=str((DATA/'train-accepted.json').relative_to(ROOT)),semantic_cases=672,training_records=1344,
        original_photo_groups=160,photo_domains=dict(portrait=80,landscape=80),new_original_groups=0,
        outcome_counts=admission['outcome_counts'],original_GIER_English_cases=97,AI_authored_cases=575,
        status='diagnostic_trial_only_not_promoted',diagnostic_choice='Final epoch1 fixed before training; no loss or checkpoint search.',
        load_method='Original Qwen full weights plus this ONE adapter; exactly the same R11 prompt/parser/editor as offline evaluation.',
        native_development_summary=read(EXP/'evaluation'/EDITOR_VERSION/'native-development'/main_id/'direct-functional-summary.json'),
        old_native_summary=read(EXP/'evaluation'/EDITOR_VERSION/'old-native'/main_id/'direct-functional-summary.json'),
        decision_summary=read(EXP/'evaluation'/EDITOR_VERSION/'decision-development'/main_id/'decision-summary.json'),
        training_protocol_sha256=sha(EXP/'training-protocol.json'),evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),
        origin_audit_sha256=sha(ROOT/'experiments/functional-v9-20261106/origin-audit.json'))
    write(target/'metadata.json',meta)
    selection=dict(at=stamp(),status='development_qualified_replication_pending' if qualified else 'failed_practical_gate',
        qualified=qualified,promoted=False,selected_model_id=None,route='language_merger',trial_default_changed=False,
        stage1_qualified_checkpoints=stage1['qualified_checkpoints'],results=stage1['results'],controlled_development_summary=stage2,
        controlled_development_status='completed' if stage2 else 'not_run_failed_stage1',
        seed_replication_started=False,acceptance_status='closed_not_evaluated',models={mid:meta},
        public_conclusion='固定第1轮模型开发通过，仍待复核与验收；默认未升级。' if qualified else '固定第1轮模型未通过功能门槛；默认未升级，验收未开封，仅供诊断试用。',
        occupied_gpu_hours_at_registration=GPUJob.consumed(read(BUDGET_EXP/'resources.json')),independent_user_feedback=0)
    write(destination,selection);write(EXP/'initial-development-selection.json',selection)
    path=ROOT/'models/catalog.json';catalog=read(path)
    assert mid not in {e.get('trial_id',e.get('id')) for e in catalog['models']}
    catalog['models'].append(dict(id=str(target.relative_to(ROOT/'models')),trial_id=mid,canonical_path=meta['adapter'],
        display_name=meta['display_name'],kind='lora_adapter',parent=meta['parent'],runtime_family='functional_v11',origin='base',
        status=meta['status'],metadata=meta['adapter']+'/metadata.json',adapter_sha256=digest))
    catalog['changed_on']='2026-11-06';write(path,catalog)
    print('Registered R11 fixed endpoint; qualified',qualified,'default unchanged, acceptance closed.')


if __name__=='__main__':main()

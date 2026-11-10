"""Register a failed route for diagnostic use, never as a qualified winner."""
import shutil
from functional_v1.common import ROOT,read,write,sha,stamp,GPUJob,EXP as BUDGET_EXP
from .runtime import EXP,DATA
from functional_v11.outcomes import EDITOR_VERSION,SCHEMA_VERSION


def main():
    path=EXP/'development-selection.json';assert not path.exists()
    p=read(EXP/'training-protocol.json');a=read(DATA/'admission.json');stage1=read(EXP/'stage1-decision.json')
    assert all(sha(ROOT/name)==h for name,h in p['code_hashes'].items())
    base=EXP/'evaluation'/EDITOR_VERSION;eligible=stage1['qualified_checkpoints']
    assert all(mid in stage1['results'] for mid in ('visual-late4-e1','visual-late4-e2'))
    stage2={}
    for mid in eligible:
        summary=read(base/'development'/mid/'direct-functional-summary.json')
        assert summary['status']=='completed' and not summary['missing_unique_images']
        stage2[mid]=summary
        assert not (summary['fully_completed_both_languages']>=81 and summary['major_quality_issues']==0),'Qualified route requires replication and lock workflow.'
    # There is no qualified champion. Expose the earliest eligible checkpoint,
    # or simply epoch1 if neither qualified, and label that diagnostic decision.
    epoch=min(int(mid.rsplit('e',1)[1]) for mid in eligible) if eligible else 1
    mid_eval=f'visual-late4-e{epoch}'
    native=read(base/'native-development'/mid_eval/'direct-functional-summary.json')
    old=read(base/'old-native'/mid_eval/'direct-functional-summary.json')
    controls=read(base/'decision-development'/mid_eval/'decision-summary.json')
    run=EXP/'runs/visual_late4-seed-20261127';state=read(run/'status.json')
    assert state['status']=='completed' and state['step']==926
    assert (state['changed_language_tensors'],state['changed_visual_merger_tensors'],state['changed_visual_block_tensors'])==(112,4,40)
    checkpoint=run/f'checkpoint-{epoch*463}'
    target=ROOT/'models/from-functional-v11/sft-visual-late4/seed-01';assert not target.exists();target.mkdir(parents=True)
    for name in ('adapter_model.safetensors','adapter_config.json'):
        shutil.copy2(checkpoint/name,target/name);assert sha(target/name)==sha(checkpoint/name)
    mid='functional-v14-visual-late4-20261127';digest=sha(target/'adapter_model.safetensors')
    controlled=stage2.get(mid_eval)
    result_text=f'原始开发{native["fully_completed_both_languages"]}/20'
    result_text+=f'、受控{controlled["fully_completed_both_languages"]}/100' if controlled else '，未进入受控开发'
    display=f'第十四轮 · 视觉末四层 · 第{epoch}轮（未通过）'
    meta=dict(model_id=mid,evaluation_id=mid_eval,display_name=display,
        trial_status=f'实验模型，AI辅助评审；{result_text}，未过功能门槛，仅供诊断试用。未知照片可能漏项、损伤画质或误拒绝；拒绝时不返回编辑结果。',
        adapter=str(target.relative_to(ROOT)),adapter_sha256=digest,origin='base',full_weight_origin=p['origin_identity'],
        parent=p['parent_adapter'],parent_sha256=p['parent_sha256'],parent_status=p['parent_status'],
        initial_adapter='One adapter;116 parent tensors exact including4 trained merger;40 new LoRA tensors in visual blocks28..31, new B zero.',
        trainable_tensors=156,trainable_parameters=3371648,visual_blocks=[28,29,30,31],
        checkpoint=str(checkpoint.relative_to(ROOT)),epoch=epoch,step=463*epoch,runtime_family='functional_v14',training_route='visual_late4',
        editor=EDITOR_VERSION,schema=SCHEMA_VERSION,dataset_sha256=p['dataset_sha256'],cases_sha256=p['cases_sha256'],
        actual_training_data=str((DATA/'train-accepted.json').relative_to(ROOT)),semantic_cases=925,training_records=1850,original_photo_groups=240,
        photo_domains=a['photo_domains'],new_original_groups=0,new_semantic_cases=0,new_independent_label_reviews=0,
        original_GIER_English_cases=154,AI_authored_cases=771,outcome_counts=a['outcomes'],status='diagnostic_trial_only_not_promoted',
        diagnostic_choice='Earliest stage1-eligible checkpoint, or epoch1 if neither eligible. All eligible checkpoints failed controlled gates; not a qualified champion or default recommendation.',
        load_method='Audited original Qwen full weights plus this ONE adapter, exact frozen R11 prompt/parser/editor and R14 runtime as offline.',
        native_development_summary=native,old_native_summary=old,decision_summary=controls,controlled_development_summary=controlled,
        training_protocol_sha256=sha(EXP/'training-protocol.json'),evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),
        origin_audit_sha256=sha(ROOT/'experiments/functional-v9-20261106/origin-audit.json'))
    write(target/'metadata.json',meta)
    stage_text='；受控门槛未通过' if eligible else '；两个检查点均未过原始请求初筛'
    public='第十四轮'+result_text+stage_text+'。未证实总体功能改善，仅供诊断试用；默认未升级，复核未启动，最终验收未开封。'
    result=dict(at=stamp(),status='failed_practical_gate',qualified=False,promoted=False,selected_model_id=None,route=None,
        trial_default_changed=False,models={mid:meta},stage1=stage1,controlled_summaries=stage2,
        parent_controlled_summary=read(base/'development/parent-frozen/direct-functional-summary.json'),
        matched_narrow_controlled_summary=read(base/'development/matched-narrow-e1/direct-functional-summary.json'),
        seed_replication_started=False,acceptance_status='closed_not_evaluated',independent_user_feedback=0,
        public_conclusion=public,occupied_gpu_hours_at_registration=GPUJob.consumed(read(BUDGET_EXP/'resources.json')))
    write(path,result);write(EXP/'initial-development-selection.json',result)
    catalog_path=ROOT/'models/catalog.json';catalog=read(catalog_path)
    assert mid not in {x.get('trial_id',x.get('id')) for x in catalog['models']}
    catalog['models'].append(dict(id=str(target.relative_to(ROOT/'models')),trial_id=mid,canonical_path=meta['adapter'],display_name=display,
        kind='lora_adapter',parent=meta['parent'],runtime_family='functional_v14',origin='base',status=meta['status'],metadata=meta['adapter']+'/metadata.json',adapter_sha256=digest))
    catalog['changed_on']='2026-11-06';write(catalog_path,catalog);print(public)


if __name__=='__main__':main()

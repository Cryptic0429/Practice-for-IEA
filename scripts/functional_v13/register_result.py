"""Register both failed developer-selected route checkpoints for diagnostic trial."""
import shutil
from functional_v1.common import ROOT,read,write,sha,stamp,GPUJob,EXP as BUDGET_EXP
from functional_v11.outcomes import EDITOR_VERSION,SCHEMA_VERSION
from .runtime import EXP,DATA,expected_count


def main():
    dest=EXP/'development-selection.json';assert not dest.exists()
    p=read(EXP/'training-protocol.json');a=read(DATA/'admission.json');stage1=read(EXP/'stage1-decision.json')
    for path,h in p['code_hashes'].items():assert sha(ROOT/path)==h
    base=EXP/'evaluation'/EDITOR_VERSION
    models={};results={}
    for route,label,short in [('language_merger','原微调范围','narrow'),('all_language_merger','扩展语言投影','wide')]:
        mid_eval=route.replace('_','-')+'-e1';native=read(base/'native-development'/mid_eval/'direct-functional-summary.json')
        old=read(base/'old-native'/mid_eval/'direct-functional-summary.json');control=read(base/'decision-development'/mid_eval/'decision-summary.json')
        stage2=read(base/'development'/mid_eval/'direct-functional-summary.json')
        assert mid_eval in stage1['qualified_checkpoints'] and stage2['status']=='completed' and not stage2['missing_unique_images']
        # Both epoch2 checkpoints failed the frozen stage1 gate; they cannot replace epoch1.
        assert not stage1['results'][route.replace('_','-')+'-e2']['stage1_passed']
        qualified=stage2['fully_completed_both_languages']>=81 and stage2['major_quality_issues']==0
        assert not qualified,'Qualified route requires replication/acceptance workflow, not this failed registration'
        run=EXP/'runs'/f'{route}-seed-20261127';state=read(run/'status.json')
        assert state['status']=='completed' and state['step']==926
        assert state['changed_language_tensors']+state['changed_visual_merger_tensors']==expected_count(route)
        checkpoint=run/'checkpoint-463';target=ROOT/'models/from-functional-v11/sft-real-coverage'/route/'seed-01'
        assert not target.exists();target.mkdir(parents=True)
        for name in ('adapter_config.json','adapter_model.safetensors'):
            shutil.copy2(checkpoint/name,target/name);assert sha(checkpoint/name)==sha(target/name)
        mid=f'functional-v13-real-coverage-{short}-20261127';digest=sha(target/'adapter_model.safetensors')
        meta=dict(model_id=mid,evaluation_id=mid_eval,display_name=f'第十三轮 · {label} · 第1轮（未通过）',
            trial_status='实验模型，AI辅助评审；原始开发16/20、受控73/100，未过功能门槛，仅供诊断试用。未知照片可能漏项、损伤画质或误拒绝；拒绝时不返回编辑结果。',
            adapter=str(target.relative_to(ROOT)),adapter_sha256=digest,origin='base',full_weight_origin=p['origin_identity'],
            parent=p['parent_adapter'],parent_sha256=p['parent_sha256'],parent_status=p['parent_status'],
            initial_adapter='One adapter, exact116 trained parent tensors including4 merger tensors copied; broader route adds280 language tensors with zero B.',
            checkpoint=str(checkpoint.relative_to(ROOT)),epoch=1,step=463,runtime_family='functional_v13',training_route=route,
            editor=EDITOR_VERSION,schema=SCHEMA_VERSION,dataset_sha256=p['dataset_sha256'],cases_sha256=p['cases_sha256'],
            actual_training_data=str((DATA/'train-accepted.json').relative_to(ROOT)),semantic_cases=925,training_records=1850,
            original_photo_groups=240,photo_domains=a['photo_domains'],new_original_groups=80,new_semantic_cases=253,
            original_GIER_English_cases=154,AI_authored_cases=771,outcome_counts=a['outcomes'],
            status='diagnostic_trial_only_not_promoted',
            diagnostic_choice='Epoch1 was the only stage1-eligible checkpoint in this route; epoch2 failed. Epoch1 then failed controlled development. This is not a qualified champion selection.',
            load_method='Audited original Qwen full weights plus this ONE adapter; exactly frozen R11 prompt/parser/editor and R13 route as offline evaluation.',
            native_development_summary=native,old_native_summary=old,decision_summary=control,controlled_development_summary=stage2,
            training_protocol_sha256=sha(EXP/'training-protocol.json'),evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),
            origin_audit_sha256=sha(ROOT/'experiments/functional-v9-20261106/origin-audit.json'))
        write(target/'metadata.json',meta);models[mid]=meta
        results[mid_eval]=dict(stage1=stage1['results'][mid_eval],controlled_complete=stage2['fully_completed_both_languages'],
            controlled_major=stage2['major_quality_issues'],all_development_gates_passed=False)
    selection=dict(at=stamp(),status='failed_controlled_practical_gate',qualified=False,promoted=False,
        selected_model_id=None,route=None,trial_default_changed=False,models=models,results=results,
        stage1_qualified_checkpoints=stage1['qualified_checkpoints'],controlled_development_status='completed_both_routes_failed',
        seed_replication_started=False,acceptance_status='closed_not_evaluated',independent_user_feedback=0,
        parent_controlled_summary=read(base/'development/parent-frozen/direct-functional-summary.json'),
        public_conclusion='第十三轮两条路线原始开发均16/20，但受控均73/100，父模型77/100；训练未证实总体功能改善。两者仅供诊断试用，默认未升级，复核未启动，最终验收未开封。',
        occupied_gpu_hours_at_registration=GPUJob.consumed(read(BUDGET_EXP/'resources.json')))
    write(dest,selection);write(EXP/'initial-development-selection.json',selection)
    path=ROOT/'models/catalog.json';catalog=read(path)
    for mid,meta in models.items():
        assert mid not in {e.get('trial_id',e.get('id')) for e in catalog['models']}
        catalog['models'].append(dict(id=str((ROOT/meta['adapter']).relative_to(ROOT/'models')),trial_id=mid,
            canonical_path=meta['adapter'],display_name=meta['display_name'],kind='lora_adapter',parent=meta['parent'],
            runtime_family='functional_v13',origin='base',status=meta['status'],metadata=meta['adapter']+'/metadata.json',adapter_sha256=meta['adapter_sha256']))
    catalog['changed_on']='2026-11-06';write(path,catalog)
    print(selection['public_conclusion'])


if __name__=='__main__':main()

"""Guarded diagnostic registration if both declared epochs fail stage1."""
import shutil
from safetensors import safe_open
from functional_v1.common import ROOT,read,write,sha,stamp,GPUJob,EXP as BUDGET
from .runtime import EXP,DATA
from .outcomes import EDITOR_VERSION,SCHEMA_VERSION

def main():
    target=EXP/'development-selection.json';assert not target.exists()
    p=read(EXP/'training-protocol.json');a=read(DATA/'admission.json')
    for name,digest in p['code_hashes'].items():assert sha(ROOT/name)==digest,name
    stages=[read(EXP/f'epoch{epoch}-stage1.json') for epoch in (1,2)]
    for epoch,stage in enumerate(stages,1):
        assert list(stage['results'])==[f'photometric-bridge-e{epoch}']
        assert not stage['qualified_checkpoints'],'A stage1-eligible model needs actual controlled review, not this failure shortcut'
        assert stage['training_protocol_sha256']==sha(EXP/'training-protocol.json')
    run=EXP/'runs/photometric_bridge-seed-20261129';state=read(run/'status.json')
    assert state['status']=='completed' and state['step']==1002
    assert (state['changed_language_tensors'],state['changed_merger_tensors'],state['changed_bridge_tensors'])==(112,4,4)
    checkpoint=run/'checkpoint-1002';weights=checkpoint/'adapter_model.safetensors'
    assert sha(weights)==state['adapter_sha256']
    config=read(checkpoint/'adapter_config.json');assert config['modules_to_save']==['stat_bridge']
    with safe_open(str(weights),framework='pt',device='cpu') as saved:
        assert len(saved.keys())==120
        bridge={k:list(saved.get_slice(k).get_shape()) for k in saved.keys() if '.stat_bridge.' in k}
        assert len(bridge)==4
        assert sorted(bridge.values(),key=str)==sorted([[128,16],[128],[3584,128],[3584]],key=str)
    dest=ROOT/'models/from-functional-v15/sft-photometric-bridge/seed-01';dest.mkdir(parents=True,exist_ok=False)
    for name in ('adapter_model.safetensors','adapter_config.json'):
        shutil.copy2(checkpoint/name,dest/name);assert sha(dest/name)==sha(checkpoint/name)
    eid='photometric-bridge-e2';base=EXP/'evaluation'/EDITOR_VERSION
    native=read(base/'native-development'/eid/'direct-functional-summary.json')
    old=read(base/'old-native'/eid/'direct-functional-summary.json')
    control=read(base/'decision-development'/eid/'decision-summary.json')
    n,o,c=native['fully_completed_both_languages'],old['fully_completed_both_languages'],control['passed_both_languages']
    major=native['major_quality_issues']
    mid='functional-v19-photometric-bridge-20261129';display='第十九轮 · 明暗与色彩输入分支 · 第2轮（未通过）'
    origin=read(ROOT/'experiments/functional-v15-20261106/training-protocol.json')['origin_identity']
    meta=dict(model_id=mid,evaluation_id=eid,display_name=display,
        trial_status=f'实验模型，AI辅助评审；原始开发{n}/20、旧回归{o}/6、保持与拒绝{c}/20，原始开发标注重大副作用{major}例；未通过功能门槛。固定末轮仅供诊断比较；拒绝时不返回结果图。',
        adapter=str(dest.relative_to(ROOT)),adapter_sha256=sha(weights),origin='base',full_weight_origin=origin,
        parent=p['parent_adapter'],parent_sha256=p['parent_sha256'],checkpoint=str(checkpoint.relative_to(ROOT)),
        epoch=2,step=1002,seed=20261129,runtime_family='functional_v19',training_route='photometric_bridge',
        inference_implementation='Exact frozen R19 input-RGB bridge; original image/literal request, R15 prompt/parser/editor. Requires R19 loader, not plain R15.',
        trainable_tensors=120,LoRA_parameters=2674688,bridge_parameters=464512,bridge_tensor_shapes=bridge,
        editor=EDITOR_VERSION,schema=SCHEMA_VERSION,dataset_sha256=p['dataset_sha256'],cases_sha256=p['cases_sha256'],
        input_features_sha256=p['input_features_sha256'],photometry_source_sha256=sha(ROOT/'scripts/functional_v19/photometry.py'),
        bridge_source_sha256=sha(ROOT/'scripts/functional_v19/bridge.py'),actual_training_data=str((DATA/'train-accepted.json').relative_to(ROOT)),
        semantic_cases=925,training_records=1850,original_photo_groups=240,photo_domains=a['domains'],
        presentations_per_epoch=2002,epochs=2,repeated_positive_semantics=19,negative_forward=False,
        new_original_groups=0,new_positive_parameter_labels=0,original_GIER_English_cases=154,AI_authored_cases=771,
        status='diagnostic_trial_only_not_promoted',
        diagnostic_choice='Both full epochs failed stage1. Fixed final epoch2 exposed for comparison, no checkpoint winner or default promotion.',
        native_development_summary=native,old_native_summary=old,decision_summary=control,controlled_development_summary=None,
        training_protocol_sha256=sha(EXP/'training-protocol.json'),evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),
        origin_audit_sha256=sha(ROOT/'experiments/functional-v9-20261106/origin-audit.json'))
    write(dest/'metadata.json',meta)
    public=f'第十九轮两个检查点均未通过原始任务门槛；固定末轮原始开发{n}/20、旧回归{o}/6、保持与拒绝{c}/20、原始开发重大副作用{major}例。未证实实用性改善，默认未升级，受控开发、复核和最终验收未启动。'
    result=dict(at=stamp(),status='failed_practical_gate',qualified=False,promoted=False,selected_model_id=None,route=None,
        trial_default_changed=False,models={mid:meta},stages=stages,controlled_evaluation_started=False,seed_replication_started=False,
        acceptance_status='closed_not_evaluated',independent_user_feedback=0,public_conclusion=public,
        occupied_gpu_hours_at_registration=GPUJob.consumed(read(BUDGET/'resources.json')))
    write(target,result);write(EXP/'initial-development-selection.json',result)
    catalog=read(ROOT/'models/catalog.json');assert mid not in {x.get('trial_id',x.get('id')) for x in catalog['models']}
    catalog['models'].append(dict(id=meta['adapter'].removeprefix('models/'),trial_id=mid,canonical_path=meta['adapter'],display_name=display,
        kind='lora_adapter',parent=meta['parent'],runtime_family='functional_v19',origin='base',status=meta['status'],metadata=meta['adapter']+'/metadata.json',adapter_sha256=meta['adapter_sha256']))
    catalog['changed_on']='2026-11-07';write(ROOT/'models/catalog.json',catalog);print(public)

if __name__=='__main__':main()

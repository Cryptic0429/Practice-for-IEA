"""Expose both fixed-final failed routes for inspection, without choosing a winner."""
import shutil
from functional_v1.common import ROOT, read, write, sha, stamp, GPUJob, EXP as BUDGET
from .runtime import EXP, DATA, ROUTES, predict, previous
from .outcomes import EDITOR_VERSION, SCHEMA_VERSION


def main():
    target=EXP/'development-selection.json'
    assert not target.exists()
    p=read(EXP/'training-protocol.json'); a=read(DATA/'admission.json')
    assert predict is previous.predict
    for name,digest in p['code_hashes'].items(): assert sha(ROOT/name)==digest
    stages=[read(EXP/f'epoch{epoch}-stage1.json') for epoch in (1,2)]
    assert all(not s['qualified_checkpoints'] for s in stages)
    assert len([r for s in stages for r in s['results']])==4
    models={}
    for route,slug,label in [('counterfactual_rank','sft-counterfactual-ranking','图像偏好排序'),('matched_sft','sft-repeated-positive-control','同预算正例对照')]:
        run=EXP/'runs'/f'{route}-seed-20261129'; state=read(run/'status.json')
        assert state['status']=='completed' and state['step']==1002
        assert (state['changed_language_tensors'],state['changed_visual_merger_tensors'])==(112,4)
        checkpoint=run/'checkpoint-1002'
        dest=ROOT/'models/from-functional-v15'/slug/'seed-01'; dest.mkdir(parents=True,exist_ok=False)
        for name in ('adapter_model.safetensors','adapter_config.json'):
            shutil.copy2(checkpoint/name,dest/name); assert sha(dest/name)==sha(checkpoint/name)
        digest=sha(dest/'adapter_model.safetensors'); assert digest==state['adapter_sha256']
        eid=('counterfactual-rank' if route=='counterfactual_rank' else 'matched-sft')+'-e2'
        base=EXP/'evaluation'/EDITOR_VERSION
        native=read(base/'native-development'/eid/'direct-functional-summary.json')
        old=read(base/'old-native'/eid/'direct-functional-summary.json')
        control=read(base/'decision-development'/eid/'decision-summary.json')
        mid='functional-v18-'+('counterfactual-ranking' if route=='counterfactual_rank' else 'matched-sft')+'-20261129'
        display='第十八轮 · '+label+' · 第2轮（未通过）'
        origin=read(ROOT/'experiments/functional-v15-20261106/training-protocol.json')['origin_identity']
        meta=dict(model_id=mid,evaluation_id=eid,display_name=display,
            trial_status='实验模型，AI辅助评审；原始开发16/20、旧回归3/6，未通过功能门槛。固定末轮仅供对照试用，不是获选模型；拒绝时不返回结果图。',
            adapter=str(dest.relative_to(ROOT)),adapter_sha256=digest,origin='base',full_weight_origin=origin,
            parent=p['parent_adapter'],parent_sha256=p['parent_sha256'],checkpoint=str(checkpoint.relative_to(ROOT)),
            epoch=2,step=1002,seed=20261129,runtime_family='functional_v18',training_route=route,
            inference_implementation='Exact functional_v15 load/predict aliases; same one-adapter architecture, literal prompt/parser/editor. R18 child supplies R18 provenance.',
            trainable_tensors=116,trainable_parameters=2674688,editor=EDITOR_VERSION,schema=SCHEMA_VERSION,
            dataset_sha256=p['dataset_sha256'],cases_sha256=p['cases_sha256'],actual_training_data=str((DATA/'train-accepted.json').relative_to(ROOT)),
            semantic_cases=925,training_records=1850,original_photo_groups=240,photo_domains=a['source_domains'],
            preference_semantics=19,preference_language_records=38,presentations_per_epoch=2002,epochs=2,
            new_original_groups=0,new_positive_parameter_labels=0,original_GIER_English_cases=154,AI_authored_cases=771,
            status='diagnostic_trial_only_not_promoted',diagnostic_choice='Both epochs of both routes failed stage1. Fixed final epoch2 exposed for comparison; no checkpoint winner or development promotion.',
            native_development_summary=native,old_native_summary=old,decision_summary=control,controlled_development_summary=None,
            training_protocol_sha256=sha(EXP/'training-protocol.json'),evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),
            origin_audit_sha256=sha(ROOT/'experiments/functional-v9-20261106/origin-audit.json'))
        write(dest/'metadata.json',meta);models[mid]=meta
    public='第十八轮两路共四个检查点均为原始开发16/20、旧回归3/6、保持与拒绝控制20/20；未通过功能门槛。训练图上的参数区分改善没有转化为开发集改善；默认未升级，复核与最终验收未启动。'
    result=dict(at=stamp(),status='failed_practical_gate',qualified=False,promoted=False,selected_model_id=None,route=None,
        trial_default_changed=False,models=models,stages=stages,controlled_evaluation_started=False,seed_replication_started=False,
        acceptance_status='closed_not_evaluated',independent_user_feedback=0,public_conclusion=public,
        occupied_gpu_hours_at_registration=GPUJob.consumed(read(BUDGET/'resources.json')))
    write(target,result);write(EXP/'initial-development-selection.json',result)
    catalog=read(ROOT/'models/catalog.json')
    for mid,m in models.items():
        assert mid not in {x.get('trial_id',x.get('id')) for x in catalog['models']}
        catalog['models'].append(dict(id=m['adapter'].removeprefix('models/'),trial_id=mid,canonical_path=m['adapter'],display_name=m['display_name'],
            kind='lora_adapter',parent=m['parent'],runtime_family='functional_v18',origin='base',status=m['status'],metadata=m['adapter']+'/metadata.json',adapter_sha256=m['adapter_sha256']))
    catalog['changed_on']='2026-11-07';write(ROOT/'models/catalog.json',catalog)
    print(public)


if __name__=='__main__':main()

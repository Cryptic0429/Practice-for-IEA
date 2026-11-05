"""Apply frozen functional gates; expose final-epoch diagnostic adapters without promotion."""
import json,shutil
from functional_v1.common import ROOT,read,write,sha,stamp,GPUJob
from .source_bank import EXP,DATA
from .evaluate import EVAL

def main():
    dest=EXP/'development-selection.json';assert not dest.exists()
    ids=['raw-base','current-default','round2-parent']+[f'{r}-e{e}' for r in ('base','continued') for e in (1,2,3,4)]
    scores={s:{mid:read(EVAL/s/mid/'direct-functional-summary.json') for mid in ids} for s in ('native-development','old-native')}
    assert all(x['status']=='completed' and not x['missing_unique_images'] for group in scores.values() for x in group.values())
    gate=read(EXP/'evaluation-protocol.json')['stage1'];qualified=[]
    for mid in ids[3:]:
        n,o=scores['native-development'][mid],scores['old-native'][mid]
        if n['fully_completed_both_languages']>=gate['minimum_native_development_qualified'] and o['fully_completed_both_languages']>=gate['minimum_old6_qualified'] and n['major_quality_issues']==o['major_quality_issues']==gate['maximum_major_cases']:qualified.append(mid)
    assert not qualified,'Qualified checkpoints require controlled development and conditional seed replication; do not freeze a failed-pilot report.'
    p=read(EXP/'training-protocol.json');a=read(DATA/'admission.json')
    runs={route:read(EXP/'runs'/f'{route}-seed-20261110/status.json') for route in ('base','continued')}
    assert all(r['status']=='completed' and r['step']==920 and r['weights_changed'] for r in runs.values())
    audits=[read(EXP/'runs'/f'{r}-seed-20261110/token-audit.json') for r in ('base','continued')]
    assert audits[0]==audits[1] and len(audits[0])==920
    result=dict(status='native_coverage_pilot_failed_practical_gate',at=stamp(),qualified=False,promoted=False,selected_model_id=None,
        native_development_results=scores['native-development'],old_native_results=scores['old-native'],models={},
        stage1_qualified_checkpoints=[],controlled_development_status='not_run_all_checkpoints_failed_predeclared_stage1',
        seed_replication_started=False,acceptance_status='closed_not_evaluated',trial_default_changed=False,
        reason='Best trained native development15/20 retains3 major cases; all trained checkpoints remain3/6 on old regressions. None meet16/20,4/6,zero-major gates. No L1 or parameter error used.',
        diagnostic_policy='Each route final fourth epoch is offered as a reproducible end-of-training diagnostic, not a qualified winner or an optimistic checkpoint pick. All eight checkpoint results stay visible. Default remains unchanged.',
        training_protocol_sha256=sha(EXP/'training-protocol.json'),evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),
        token_audits_identical=True,admission=a,training_runs=runs,independent_user_feedback=0)
    for route in ('base','continued'):
        mid=route+'-e4';source=EXP/'runs'/f'{route}-seed-20261110'/'checkpoint-920'
        assert sha(source/'adapter_model.safetensors')==runs[route]['adapter_sha256']
        folder=ROOT/'models'/('from-base' if route=='base' else 'from-functional-v2')/'sft-native-coverage'/'seed-01'
        folder.mkdir(parents=True,exist_ok=False)
        for f in ('adapter_config.json','adapter_model.safetensors'):shutil.copy2(source/f,folder/f)
        item=dict(model_id=f'functional-v6-{route}-20261110',evaluation_id=mid,
            display_name='第六轮 · '+('基础起点' if route=='base' else '继续训练')+' · 第4轮（未通过）',
            trial_status='实验模型，AI辅助评审；本轮功能门槛未通过，仍有高反差亮部损失、噪声放大和力度/复合要求失败。仅供实际试用和诊断。',
            adapter=str(folder.relative_to(ROOT)),adapter_sha256=sha(folder/'adapter_model.safetensors'),
            parent='models/Qwen2.5-VL-7B-Instruct' if route=='base' else p['parent'],
            initial_adapter_sha256=None if route=='base' else p['parent_sha256'],checkpoint=str(source.relative_to(ROOT)),epoch=4,step=920,
            runtime_family='functional_v1',runtime_sha256=sha(ROOT/'scripts/functional_v1/runtime.py'),editor=p['editor'],
            dataset_sha256=p['dataset_sha256'],cases_sha256=p['cases_sha256'],training_records=920,semantic_cases=460,original_photo_groups=160,
            original_native_request_semantics=100,AI_controlled_semantics=360,photo_domains={'portrait':80,'landscape':80},
            source_directions='Original English and paired AI Chinese translations grouped; no repeats, reference image, review or target leakage.',
            status='diagnostic_trial_only_not_promoted',diagnostic_choice=result['diagnostic_policy'],
            load_method='Original Qwen base plus this ONE adapter. Never stack historical parent adapters.',
            native_development_summary=scores['native-development'][mid],old_native_summary=scores['old-native'][mid],
            user_approval=False,at=stamp())
        write(folder/'metadata.json',item);result['models'][item['model_id']]=item
    ledger=read(ROOT/'experiments/functional-v1-20261104/resources.json');result['resource_snapshot']=dict(at=stamp(),occupied_gpu_hours=GPUJob.consumed(ledger),limit_gpu_hours=16)
    write(dest,result)
    write(EXP/'training-verification.json',dict(at=stamp(),status='passed',token_audits_identical=True,records_each=920,
        updates_each=920,epochs_each=4,epoch_checkpoint_steps=[230,460,690,920],trainable_tensors=112,finite_gradients_checked_each_optimizer_step=True,
        data_and_runtime_match_locked_protocol=True,weights_changed_both_routes=True,final_matches_epoch4_both_routes=True,
        note='Loss and finite gradients verify training mechanics, not functional usefulness.'))
    print(json.dumps(dict(status=result['status'],models=list(result['models']),resources=result['resource_snapshot'])))

if __name__=='__main__':main()

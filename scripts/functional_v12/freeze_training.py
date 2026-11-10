"""Freeze a prompt-only control and one matched-manual short training route."""
import copy
import hashlib
from functional_v1.common import ROOT,read,write,sha,stamp,GPUJob,EXP as BUDGET_EXP
from .runtime import EXP,DATA,PARENT,PARENT_SHA,verify_origin
from .outcomes import system_prompt,EDITOR_VERSION,SCHEMA_VERSION


def main():
    assert not (EXP/'training-protocol.json').exists()
    consumed=GPUJob.consumed(read(BUDGET_EXP/'resources.json'))
    assert 16-consumed>1.1, 'Keep the full conditional route inside the original cap'
    a=read(DATA/'admission.json');assert a['semantic_cases']==250 and a['source_groups']==60
    transfer=read(EXP/'runtime-transfer-audit.json')
    assert transfer['parent_sha256']==PARENT_SHA and transfer['results'][0]['parent_tensors_copied_exactly']==116
    cache=read(EXP/'batch-cache.json')
    assert cache['rows']==500 and cache['dataset_sha256']==sha(DATA/'train-accepted.json')
    assert cache['cases_sha256']==sha(DATA/'cases.json') and sha(ROOT/cache['cache_path'])==cache['cache_sha256']
    for file in ('target-replay.json','source-isolation.json','input-audit.json','manual-audit.json'):
        assert read(EXP/file)['status']=='passed'
    old=read(ROOT/'experiments/functional-v11-20261106/training-protocol.json')
    for path,digest in old['code_hashes'].items():assert sha(ROOT/path)==digest,'Frozen dependency changed'
    evaluation=copy.deepcopy(read(ROOT/'experiments/functional-v11-20261106/evaluation-protocol.json'))
    for item in evaluation['case_files'].values():assert sha(ROOT/item['path'])==item['sha256']
    evaluation.update(at=stamp(),status='round12_cases_gates_and_general_manual_frozen_before_GPU_forward',
        comparisons=['Frozen R11 weights with the new manual: parent-new-manual, actual model inference, no new training.',
            'Same exact R11 parent after one250-case new-manual epoch, evaluated with that identical manual.',
            'Frozen R11 original-manual15/20, old4/6, controls20/20 retained to distinguish prompt and training contributions.'],
        selection=dict(checkpoint='Only final epoch1 step125 eligible; fixed before training. No loss or epoch search.',
            eligibility='Native>=16/20 and old>=4/6 and controls20/20 with0 marked major, then controlled>=81/100 with0 marked major. A feasible-task refusal or illegal output is uncompleted. New trained candidate must not worsen functional completion or severe effects versus the same-parent new-manual control.',
            rank=['Single predefined trained route. Report prompt-only contribution separately; no false training-gain claim.']),
        limitations=['Development images repeatedly viewed; general manual designed after observed error classes, not a pristine heldout prompt test.',
            'All image quality/completion assessments AI-assisted, not user approval. Historical unresolved U retained.',
            'No heldout image, reference parameter, original-image statistic, user upload or feedback enters training/manual.'],
        new_training_started=False,actual_user_feedback=0)
    write(EXP/'evaluation-protocol.json',evaluation)
    files=['scripts/functional_v1/common.py','scripts/functional_v1/editor.py','scripts/functional_v1/runtime.py',
        'scripts/functional_v7/runtime.py','scripts/functional_v9/runtime.py','scripts/functional_v11/editor.py','scripts/functional_v11/outcomes.py']
    files += ['scripts/functional_v12/'+name for name in ('outcomes.py','runtime.py','train.py','evaluate.py','admit_training.py',
        'prepare_batches.py','audit_runtime.py','audit_manual.py','freeze_training.py')]
    write(EXP/'training-protocol.json',dict(at=stamp(),status='actual_data_runtime_optimizer_and_functional_gates_frozen_before_training',
        hypothesis='Clarify signed-delta operator semantics and one-way sepia generally, retain the exact R11 editor/parser, compare prompt-only frozen-parent inference with a small matched-manual correction. Never repair illegal JSON or specialize runtime rules to evaluation cases.',
        routes=['language_merger'],origin='base',origin_identity=verify_origin(),
        parent_adapter=str(PARENT.relative_to(ROOT)),parent_sha256=PARENT_SHA,
        parent_status='R11 failed research candidate15/20, matching editor-only control; not a qualified champion',
        prompt_only_control=dict(model_id='parent-new-manual',adapter=str(PARENT.relative_to(ROOT)),adapter_sha256=PARENT_SHA,
            actual_training_data_sha256=old['dataset_sha256'],new_training=False,model_inference=True),
        training=dict(rows=500,epochs=1,steps_per_epoch=125,steps=125,learning_rate=2e-5,effective_batch=4,
            rank=8,alpha=16,dropout=0.,seed=20261125,replicate_seed=20261126,per_device_batch=1,gradient_accumulation=4,
            image_max_pixels=262144,max_sequence_tokens=2048,quantization='NF4, double quantization, BF16 compute',
            optimizer='adamw_torch',weight_decay=0.,scheduler='cosine',warmup_ratio=.05,max_grad_norm=1.,
            save='Final complete epoch1 only',assistant_only_decision_JSON_supervision=True,
            language_trainable_tensors=112,merger_trainable_tensors=116,exact_parent_copy=116,stacked_adapters=False,
            training_and_trained_model_inference_prompt_identical=True),
        dataset_sha256=sha(DATA/'train-accepted.json'),cases_sha256=sha(DATA/'cases.json'),admission_sha256=sha(DATA/'admission.json'),
        input_audit_path=str((EXP/'input-audit.json').relative_to(ROOT)),input_audit_sha256=sha(EXP/'input-audit.json'),
        target_replay_path=str((EXP/'target-replay.json').relative_to(ROOT)),target_replay_sha256=sha(EXP/'target-replay.json'),
        batch_cache_path=cache['cache_path'],batch_cache_sha256=cache['cache_sha256'],
        evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),code_hashes={p:sha(ROOT/p) for p in files},
        editor=EDITOR_VERSION,schema=SCHEMA_VERSION,runtime='functional_v12.runtime',
        system_prompt_sha256=hashlib.sha256(system_prompt().encode()).hexdigest(),source_counts=a,
        resource=dict(shared_cap_GPU_hours=16,consumed_before_round12=consumed,remaining_at_freeze=16-consumed,
            allowed_GPUs=[0,2,3],preserve_GPU1=True,estimated_main_training=.12,estimated_main_and_prompt_control_stage1=.15,
            conditional_replication_controlled_acceptance_trial_reserve=.8,
            on_limit='Save and stop owned experimental jobs before original16GPUh; watchdog15.9. Never extend budget or touch GPU1.')))
    print('Frozen250semantic500rows/125steps, new manual matched training/inference and prompt-only control; GPUh remaining',16-consumed)


if __name__=='__main__':main()

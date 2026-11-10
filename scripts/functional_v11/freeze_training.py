"""Freeze one correction route and editor-only control before new GPU forwards."""
import copy
import hashlib
from functional_v1.common import ROOT, read, write, sha, stamp, GPUJob, EXP as BUDGET_EXP
from .runtime import EXP, DATA, PARENT, PARENT_SHA, verify_origin
from .outcomes import system_prompt, EDITOR_VERSION, SCHEMA_VERSION


def main():
    assert not (EXP/'training-protocol.json').exists()
    consumed=GPUJob.consumed(read(BUDGET_EXP/'resources.json'))
    assert 16-consumed > 1.4, 'Reserve conditional replication, development, acceptance and trial'
    admission=read(DATA/'admission.json')
    assert admission['semantic_cases']==672 and admission['source_groups']==160
    transfer=read(EXP/'runtime-transfer-audit.json')
    assert transfer['parent_sha256']==PARENT_SHA
    assert transfer['results'][0]['parent_tensors_copied_exactly']==116
    cache=read(EXP/'batch-cache.json')
    assert cache['rows']==1344 and cache['dataset_sha256']==sha(DATA/'train-accepted.json')
    assert cache['cases_sha256']==sha(DATA/'cases.json') and sha(ROOT/cache['cache_path'])==cache['cache_sha256']
    for file in ('target-replay.json','source-isolation.json','input-audit.json','editor-invariant-tests.json','outcome-contract-check.json'):
        assert read(EXP/file)['status']=='passed'
    previous=read(ROOT/'experiments/functional-v10-20261106/training-protocol.json')
    for path,digest in previous['code_hashes'].items():assert sha(ROOT/path)==digest,'Frozen dependency changed'
    evaluation=copy.deepcopy(read(ROOT/'experiments/functional-v10-20261106/evaluation-protocol.json'))
    for item in evaluation['case_files'].values():assert sha(ROOT/item['path'])==item['sha256']
    evaluation.update(at=stamp(),status='round11_cases_and_gates_frozen_before_GPU_forward',
        comparisons=['Parent fixed R10 language-merger final parameters re-executed on R11 editor, CPU-only replay; no new inference.',
            'One R11 correction from the same exact116 parent tensors, same input prompt and strict parser, new editor.',
            'Frozen R10 parent on R7 editor and original practical default retained as controls.'],
        selection=dict(checkpoint='Final epoch1 step336 fixed before training; no loss-based selection.',
            eligibility='Native>=16/20 and old>=4/6 and decision controls20/20, with0 marked major; then controlled>=81/100 with0 major. Feasible-task refusal is uncompleted. Must improve editing without new severe destruction.',
            rank=['Single predefined route, no post-hoc checkpoint choice']),
        limitations=['Development images repeatedly viewed historically; not pristine independent evaluation.',
            'Direct AI review, not human or user approval. Unresolved historical uncertainty is retained.',
            'New synthetic conditions use old training photos; no new independent original group.'],
        new_training_started=False,actual_user_feedback=0)
    write(EXP/'evaluation-protocol.json',evaluation)
    files=['scripts/functional_v1/common.py','scripts/functional_v1/editor.py','scripts/functional_v1/runtime.py',
        'scripts/functional_v7/editor_soft.py','scripts/functional_v7/outcomes.py','scripts/functional_v7/runtime.py',
        'scripts/functional_v9/runtime.py']
    files += ['scripts/functional_v11/'+p for p in ('editor.py','outcomes.py','runtime.py','train.py','evaluate.py',
        'prepare_batches.py','admit_training.py','audit_runtime.py','freeze_training.py','replay_parent.py')]
    write(EXP/'training-protocol.json',dict(at=stamp(),status='actual_data_runtime_optimizer_and_functional_gates_frozen_before_training',
        hypothesis='Correct a negative-exposure tool response that fixed white, revalidate every affected target and prior quality decline, rehearse prior directions/keep/refusal, and add10 synthetic brown-yellow correction requests to test valid cooling instead of negative sepia.',
        routes=['language_merger'],origin='base',origin_identity=verify_origin(),
        parent_adapter=str(PARENT.relative_to(ROOT)),parent_sha256=PARENT_SHA,
        parent_status='R10 failed research candidate14/20 native0 marked major; not a qualified champion',
        training=dict(rows=1344,epochs=1,steps_per_epoch=336,steps=336,learning_rate=2e-5,effective_batch=4,
            rank=8,alpha=16,dropout=0.,seed=20261123,replicate_seed=20261124,per_device_batch=1,
            gradient_accumulation=4,image_max_pixels=262144,max_sequence_tokens=2048,
            quantization='NF4, double quantization, BF16 compute',optimizer='adamw_torch',weight_decay=0.,
            scheduler='cosine',warmup_ratio=.05,max_grad_norm=1.,save='Final complete epoch1 only',
            assistant_only_decision_JSON_supervision=True,language_trainable_tensors=112,merger_trainable_tensors=116,
            exact_parent_copy=116,trained_visual_merger_parent_copy=4,stacked_adapters=False),
        dataset_sha256=sha(DATA/'train-accepted.json'),cases_sha256=sha(DATA/'cases.json'),admission_sha256=sha(DATA/'admission.json'),
        input_audit_path=str((EXP/'input-audit.json').relative_to(ROOT)),input_audit_sha256=sha(EXP/'input-audit.json'),
        target_replay_path=str((EXP/'target-replay.json').relative_to(ROOT)),target_replay_sha256=sha(EXP/'target-replay.json'),
        batch_cache_path=cache['cache_path'],batch_cache_sha256=cache['cache_sha256'],
        evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),code_hashes={p:sha(ROOT/p) for p in files},
        editor=EDITOR_VERSION,schema=SCHEMA_VERSION,runtime='functional_v11.runtime',
        system_prompt_sha256=hashlib.sha256(system_prompt().encode()).hexdigest(),source_counts=admission,
        resource=dict(shared_cap_GPU_hours=16,consumed_before_round11=consumed,remaining_at_freeze=16-consumed,
            allowed_GPUs=[0,2,3],preserve_GPU1=True,estimated_main_training=.3,estimated_stage1=.1,
            conditional_replication_and_evaluation_reserve=.9,trial_reserve=.15,
            on_limit='Save state and stop owned jobs before original shared cap; never extend budget or touch GPU1.')))
    print('Frozen single R11 final-epoch correction,672semantic1344rows; remaining GPUh',16-consumed)


if __name__=='__main__':main()

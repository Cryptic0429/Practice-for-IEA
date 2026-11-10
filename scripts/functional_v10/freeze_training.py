"""Freeze actual admitted data and two final-epoch candidates before GPU forwards."""
import copy
import hashlib
from functional_v1.common import ROOT,read,write,sha,stamp,GPUJob,EXP as BUDGET_EXP
from functional_v7.outcomes import system_prompt,EDITOR_VERSION,SCHEMA_VERSION
from .runtime import EXP,DATA,PARENT,PARENT_SHA,verify_origin


def main():
    assert not (EXP/'training-protocol.json').exists() and not (EXP/'evaluation-protocol.json').exists()
    ledger=read(BUDGET_EXP/'resources.json'); assert not any(j['stage'].startswith('v10-') for j in ledger['jobs'])
    consumed=GPUJob.consumed(ledger);assert consumed<14,'Reserve evaluation, conditional replication and trial inside16GPUh'
    identity=verify_origin();admission=read(DATA/'admission.json')
    assert admission['semantic_cases']==240 and admission['source_groups']==60 and admission['photo_domains']==dict(portrait=30,landscape=30)
    audit=read(EXP/'runtime-transfer-audit.json');assert audit['parent_sha256']==PARENT_SHA
    assert all(r['language_parent_tensors_copied_exactly']==112 and not r['stacked_adapters'] for r in audit['results'])
    cache=read(EXP/'batch-cache.json');assert cache['rows']==480 and cache['dataset_sha256']==sha(DATA/'train-accepted.json')
    assert cache['cases_sha256']==sha(DATA/'cases.json') and sha(ROOT/cache['cache_path'])==cache['cache_sha256']
    for file in ('source-replay.json','source-isolation.json','target-replay.json','input-audit.json'):
        assert read(EXP/file)['status']=='passed'
    previous=read(ROOT/'experiments/functional-v9-20261106/training-protocol.json')
    for name,digest in previous['code_hashes'].items():assert sha(ROOT/name)==digest,'Previously frozen dependency changed'
    evaluation=copy.deepcopy(read(ROOT/'experiments/functional-v9-20261106/evaluation-protocol.json'))
    for item in evaluation['case_files'].values():assert sha(ROOT/item['path'])==item['sha256']
    evaluation.update(at=stamp(),status='round10_cases_and_gates_frozen_before_GPU_forward',
        comparisons=['Fixed final epoch2 of language-only versus language+visual-merger, same parent and480rows',
            'Inherited exact parent base-e1 under identical prompt/editor:14/20 native,1major,old4/6,controls20/20',
            'Frozen practical default under same runtime:13/20 native,2major,old4/6,controls17/20',
            'Prior original-pipeline and editor-only results retained as context'],
        selection=dict(checkpoint='Final epoch2 step240 fixed before training; epoch1 saved for recovery only, not evaluated for selection.',
            eligibility='Both stage1 and stage2 gates must pass; improve over failed parent and default without major damage. Feasible-task decline is uncompleted.',
            rank=['More fully completed acceptable bilingual cases','Fewer major effects','Anonymous paired direct visual comparison','language-only if still tied']),
        limitations=['Development photos have been repeatedly viewed historically; not pristine independent evaluation.',
            'Codex direct assessment is AI review, not user approval. Independent user feedback remains zero.',
            'New synthetic photographic conditions are training-only, not real camera noise or proof of transfer.'],
        new_training_started=False,actual_user_feedback=0)
    write(EXP/'evaluation-protocol.json',evaluation)
    files=['scripts/functional_v1/common.py','scripts/functional_v1/editor.py','scripts/functional_v1/runtime.py',
        'scripts/functional_v7/editor_soft.py','scripts/functional_v7/outcomes.py','scripts/functional_v7/runtime.py',
        'scripts/functional_v9/runtime.py','scripts/functional_v10/runtime.py','scripts/functional_v10/train.py',
        'scripts/functional_v10/evaluate.py','scripts/functional_v10/admit_training.py','scripts/functional_v10/prepare_batches.py',
        'scripts/functional_v10/audit_runtime.py','scripts/functional_v10/freeze_training.py']
    write(EXP/'training-protocol.json',dict(at=stamp(),status='actual_data_runtime_optimizer_and_functional_gates_frozen_before_training',
        hypothesis='Matched photographic-condition labels force identical requests to depend on the input image; normal inputs also have opposite style requests. Compare language-only correction with a visual-merger path, without assuming synthetic-condition learning transfers to real photographs.',
        routes=['language','language_merger'],origin='base',origin_identity=identity,
        origin_audit_path='experiments/functional-v9-20261106/origin-audit.json',
        origin_audit_sha256=sha(ROOT/'experiments/functional-v9-20261106/origin-audit.json'),
        parent_adapter=str(PARENT.relative_to(ROOT)),parent_sha256=PARENT_SHA,
        parent_status='Failed research parent,14/20native with1major; not a qualified champion',
        runtime_transfer_audit_sha256=sha(EXP/'runtime-transfer-audit.json'),
        training=dict(rows=480,epochs=2,steps_per_epoch=120,steps=240,learning_rate=5e-5,effective_batch=4,
            rank=8,alpha=16,dropout=0.,seed=20261121,replicate_seed=20261122,per_device_batch=1,
            gradient_accumulation=4,image_max_pixels=262144,max_sequence_tokens=2048,
            quantization='NF4, double quantization, BF16 compute',optimizer='adamw_torch',weight_decay=0.,
            scheduler='cosine',warmup_ratio=.05,max_grad_norm=1.,save='every complete epoch; only final epoch2 evaluated',
            assistant_only_decision_JSON_supervision=True,language_trainable_tensors=112,merger_trainable_tensors=116,
            exact_language_parent_copy=True,added_merger_B_zero=True,stacked_adapters=False),
        dataset_sha256=sha(DATA/'train-accepted.json'),cases_sha256=sha(DATA/'cases.json'),admission_sha256=sha(DATA/'admission.json'),
        input_audit_path=str((EXP/'input-audit.json').relative_to(ROOT)),input_audit_sha256=sha(EXP/'input-audit.json'),
        target_replay_path=str((EXP/'target-replay.json').relative_to(ROOT)),target_replay_sha256=sha(EXP/'target-replay.json'),
        batch_cache_path=cache['cache_path'],batch_cache_sha256=cache['cache_sha256'],
        evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),code_hashes={name:sha(ROOT/name) for name in files},
        editor=EDITOR_VERSION,schema=SCHEMA_VERSION,runtime='functional_v10.runtime',system_prompt_sha256=hashlib.sha256(system_prompt().encode()).hexdigest(),
        source_counts=admission,
        resource=dict(shared_cap_GPU_hours=16,consumed_before_round10=consumed,remaining_at_freeze=16-consumed,
            allowed_GPUs=[0,2,3],preserve_GPU1=True,estimated_two_training_GPU_hours=.5,
            estimated_stage1_evaluation_GPU_hours=.25,conditional_evaluation_replication_trial_reserve_GPU_hours=1.4,
            on_limit='Save state, stop owned experimental jobs before original cap, report without expanding training.')))
    print('Round10 frozen before GPU forwards:240semantic/480rows,60groups30/30; fixed epoch2,112/116 tensors; remaining GPUh',16-consumed)

if __name__=='__main__':main()

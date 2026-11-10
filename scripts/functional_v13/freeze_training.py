"""Freeze actual data, two matched capacity routes and functional gates."""
import copy
import hashlib
import math
from functional_v1.common import ROOT,read,write,sha,stamp,GPUJob,EXP as BUDGET_EXP
from functional_v11.outcomes import system_prompt,EDITOR_VERSION,SCHEMA_VERSION
from .runtime import EXP,DATA,PARENT,PARENT_SHA,ROUTES,verify_origin


def main():
    assert not (EXP/'training-protocol.json').exists()
    ledger=read(BUDGET_EXP/'resources.json');used=GPUJob.consumed(ledger);cap=ledger['limit_gpu_hours']
    assert cap==24 and cap-used>6
    admission=read(DATA/'admission.json');assert admission['semantic_cases']==925 and admission['source_groups']==240
    cache=read(EXP/'batch-cache.json');assert cache['rows']==1850
    assert cache['dataset_sha256']==sha(DATA/'train-accepted.json') and cache['cases_sha256']==sha(DATA/'cases.json')
    assert sha(ROOT/cache['cache_path'])==cache['cache_sha256']
    for f in ('target-replay.json','input-audit.json','editor-invariant-tests.json','outcome-contract-check.json','initial-function-parity.json'):
        assert read(EXP/f)['status']=='passed'
    transfer=read(EXP/'runtime-transfer-audit.json')
    assert transfer['parent_sha256']==PARENT_SHA and transfer['runtime_sha256']==sha(ROOT/'scripts/functional_v13/runtime.py')
    assert [r['trainable_tensors'] for r in transfer['results']]==[116,396]
    old=read(ROOT/'experiments/functional-v11-20261106/training-protocol.json')
    for path,digest in old['code_hashes'].items():assert sha(ROOT/path)==digest
    evaluation=copy.deepcopy(read(ROOT/'experiments/functional-v12-20261106/evaluation-protocol.json'))
    for item in evaluation['case_files'].values():assert sha(ROOT/item['path'])==item['sha256']
    evaluation.update(at=stamp(),status='round13_cases_and_gates_frozen_before_training_and_development_outputs',
        comparisons=['Exact frozen R11 decisions, prompt/parser/editor and checkpoint retained as parent control; no new inference claimed.',
            'Same R11 parent and1850 bilingual rows;116-tensor language q/v+merger versus396-tensor full language projections+merger.',
            'Epoch1 and2 checkpoints eligible per route; no loss/L1-based choice.'],
        selection=dict(checkpoint='Epoch1 or2, same fixed seed20261127, both predefined routes.',
            eligibility='Native>=16/20, old>=4/6, controls20/20, zero marked major; then controlled>=81/100 and zero major. Must exceed parent15/20 without worsening old completion or decision controls.',
            rank=['More bilingual complete+acceptable native cases','Fewer severe side effects','Better anonymous paired visual outcome if needed','Earlier checkpoint','Narrower trainable range on exact tie']),
        limitations=['Native20 and controlled100 development inputs have been repeatedly used; this is an adaptive development comparison, not an independent new benchmark.',
            '80 added original photographs come only from the old training pool, not new user photos.',
            'Direct Codex AI review; frozen Qwen critic failed earlier calibration and cannot provide a reliable second vote. User approval remains zero.'],
        resource=dict(shared_gpu_hour_cap=cap,allowed_gpu_indices=[0,2,3],gpu1_original_service_preserved=True,
            ledger=str((BUDGET_EXP/'resources.json').relative_to(ROOT)),includes='All actual occupied model-load, training, evaluation, replication and trial GPU time'),
        new_training_started=False,actual_user_feedback=0)
    write(EXP/'evaluation-protocol.json',evaluation)
    files=set(old['code_hashes'])
    files.update(str(p.relative_to(ROOT)) for p in (ROOT/'scripts/functional_v13').glob('*.py'))
    settings=dict(rows=1850,epochs=2,steps_per_epoch=math.ceil(1850/4),steps=2*math.ceil(1850/4),
        learning_rate=5e-5,effective_batch=4,rank=8,alpha=16,dropout=0.,seed=20261127,replicate_seed=20261128,
        per_device_batch=1,gradient_accumulation=4,image_max_pixels=262144,max_sequence_tokens=2048,
        quantization='NF4, double quantization, BF16 compute',optimizer='adamw_torch',weight_decay=0.,
        scheduler='cosine',warmup_ratio=.05,max_grad_norm=1.,save='Each full epoch; select only by frozen functional gates',
        final_partial_accumulation='Each epoch ends with two examples; both routes use the same final partial update.',
        assistant_only_decision_JSON_supervision=True,exact_parent_copy=116,trained_visual_merger_parent_copy=4,
        trainable_tensors=dict(language_merger=116,all_language_merger=396),new_language_B_zero=True,stacked_adapters=False)
    write(EXP/'training-protocol.json',dict(at=stamp(),status='actual_data_runtime_optimizer_and_functional_gates_frozen_before_training',
        hypothesis='Add80 original training-pool photograph groups with explicitly verified direction/strength supervision, and compare matched narrow/broader language adaptation from the same R11 parent. No new prompt, editor or image-resolution change.',
        routes=list(ROUTES),origin='base',origin_identity=verify_origin(),parent_adapter=str(PARENT.relative_to(ROOT)),parent_sha256=PARENT_SHA,
        parent_status='R11 failed research candidate15/20 native,old4/6,controls20/20; not a qualified practical champion',training=settings,
        dataset_sha256=sha(DATA/'train-accepted.json'),cases_sha256=sha(DATA/'cases.json'),admission_sha256=sha(DATA/'admission.json'),
        input_audit_path=str((EXP/'input-audit.json').relative_to(ROOT)),input_audit_sha256=sha(EXP/'input-audit.json'),
        target_replay_path=str((EXP/'target-replay.json').relative_to(ROOT)),target_replay_sha256=sha(EXP/'target-replay.json'),
        batch_cache_path=cache['cache_path'],batch_cache_sha256=cache['cache_sha256'],
        evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),code_hashes={p:sha(ROOT/p) for p in sorted(files)},
        editor=EDITOR_VERSION,schema=SCHEMA_VERSION,runtime='functional_v13.runtime',
        system_prompt_sha256=hashlib.sha256(system_prompt().encode()).hexdigest(),source_counts=admission,
        resource=dict(shared_cap_GPU_hours=cap,consumed_at_freeze=used,remaining_at_freeze=cap-used,allowed_GPUs=[0,2,3],preserve_GPU1=True,
            main_training_max_GPU_hours=3,development_max_GPU_hours=1.5,conditional_replication_reserve=2,acceptance_and_trial_reserve=1.5,
            on_limit='Save and stop owned jobs at actual cumulative cap; no GPU1 changes.')))
    print('Frozen925semantics/1850rows,240groups,116vs396,epochs1/2; GPUh left',cap-used)


if __name__=='__main__':main()

"""Freeze one matched label/editor correction, including the cumulative budget."""
import copy
import hashlib
from functional_v1.common import ROOT,read,write,sha,stamp,GPUJob,EXP as BUDGET_EXP
from .runtime import EXP,DATA,PARENT,PARENT_SHA,ROUTES,verify_origin
from .outcomes import system_prompt,EDITOR_VERSION,SCHEMA_VERSION


def main():
    assert not (EXP/'training-protocol.json').exists()
    p13=read(ROOT/'experiments/functional-v13-20261106/training-protocol.json')
    ledger=read(BUDGET_EXP/'resources.json');used=GPUJob.consumed(ledger);cap=ledger['limit_gpu_hours']
    assert cap==24 and cap-used>4.5
    for path,digest in p13['code_hashes'].items():assert sha(ROOT/path)==digest
    for path,digest in read(EXP/'editor-only-protocol.json')['code_hashes'].items():assert sha(ROOT/path)==digest
    assert read(EXP/'runtime-audit-inheritance.json')['status']=='passed_exact_loader_alias_inheritance'
    for f in ('input-audit.json','target-replay.json','shadow-invariant-tests.json','outcome-contract-check.json'):
        assert read(EXP/f)['status']=='passed'
    control=read(EXP/'editor-control-summary.json');assert control['status']=='completed'
    assert control['GPU_hours']==0 and control['model_forward_calls']==0
    data=read(DATA/'admission.json');assert data['semantic_cases']==925 and data['unique_language_rows']==1850
    assert data['source_groups']==240 and data['reviewed_repairs']==11 and data['new_photographs']==0
    cache=read(EXP/'batch-cache.json')
    assert cache['rows']==1850 and cache['dataset_sha256']==sha(DATA/'train-accepted.json')
    assert cache['cases_sha256']==sha(DATA/'cases.json') and sha(ROOT/cache['cache_path'])==cache['cache_sha256']
    evaluation=copy.deepcopy(read(ROOT/'experiments/functional-v13-20261106/evaluation-protocol.json'))
    for item in evaluation['case_files'].values():assert sha(ROOT/item['path'])==item['sha256']
    evaluation.update(at=stamp(),status='round15_data_editor_and_functional_gates_frozen_before_optimization',
        comparisons=['Exact saved R11 parent decisions replayed under editor15, no new inference, explicit inherited/new AI uncertainty.',
            'Same R11 parent, narrow116-tensor route, literal prompt, processor, image budget, NF4 and seed as R13.',
            '925semantic/1850language rows; only11 decisions/22 labels differ from R13, zero new photographs. Shadow pixels revalidated; no claim of925 new labels.',
            'R13 narrow and R14 visual results remain frozen failed historical controls; do not tune against acceptance.'],
        selection=dict(checkpoint='Epoch1 or2 of language_merger seed20261127.',
            eligibility='Native>=16/20,old>=4/6,controls20/20,zero marked major; then controlled>=81/100 and zero major. Must exceed editor-only parent native completion; declines never count as completed feasible edits.',
            rank=['More bilingual complete+acceptable native cases','Fewer severe side effects','Better anonymous paired visual outcome if needed','Earlier checkpoint']),
        limitations=['Repeated adaptive development sets; not an independent new benchmark.',
            'Exact previous model transfer/parity audits are inherited through identical function aliases; no new GPU initial parity measurement.',
            'AI observations have same-old-image disagreements, which remain explicit and uncertain; not user approval.',
            'Quality refusals reflect finite attempts and synthetic severe noise; not proof that every possible recipe fails.'],
        resource=dict(shared_gpu_hour_cap=24,allowed_gpu_indices=[0,2,3],gpu1_original_service_preserved=True,
            ledger=str((BUDGET_EXP/'resources.json').relative_to(ROOT))),
        seed_replication='Only if every development gate passes without new major damage; seed20261128, lock before acceptance.',
        actual_user_feedback=0,new_training_started=False)
    write(EXP/'evaluation-protocol.json',evaluation)
    settings=copy.deepcopy(p13['training'])
    settings.update(trainable_tensors={'language_merger':116},new_language_B_zero=False,matched_control_seed=20261127)
    files=set(p13['code_hashes'])
    files.update(str(p.relative_to(ROOT)) for p in (ROOT/'scripts/functional_v15').glob('*.py'))
    audit_files=['runtime-audit-inheritance.json','outcome-contract-check.json','shadow-invariant-tests.json',
                 'source-isolation-inheritance.json','editor-control-summary.json','functional-review-criteria.json']
    write(EXP/'training-protocol.json',dict(at=stamp(),status='actual_data_runtime_optimizer_and_functional_gates_frozen_before_training',
        hypothesis='Correct shadow-tool colour preservation and11 visually deficient parameter labels, holding the R13 narrow optimization path fixed. A limited correction experiment, not a new dataset or assured practical model.',
        routes=list(ROUTES),origin='base',origin_identity=verify_origin(),parent_adapter=str(PARENT.relative_to(ROOT)),parent_sha256=PARENT_SHA,
        parent_status='Failed R11 research candidate; identical parameters under editor15 separately reviewed before training.',
        training=settings,dataset_sha256=sha(DATA/'train-accepted.json'),cases_sha256=sha(DATA/'cases.json'),admission_sha256=sha(DATA/'admission.json'),
        input_audit_path=str((EXP/'input-audit.json').relative_to(ROOT)),input_audit_sha256=sha(EXP/'input-audit.json'),
        target_replay_path=str((EXP/'target-replay.json').relative_to(ROOT)),target_replay_sha256=sha(EXP/'target-replay.json'),
        batch_cache_path=cache['cache_path'],batch_cache_sha256=cache['cache_sha256'],
        audit_hashes={str((EXP/f).relative_to(ROOT)):sha(EXP/f) for f in audit_files},
        evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),code_hashes={p:sha(ROOT/p) for p in sorted(files)},
        editor=EDITOR_VERSION,schema=SCHEMA_VERSION,runtime='functional_v15.runtime',
        system_prompt_sha256=hashlib.sha256(system_prompt().encode()).hexdigest(),source_counts=data,
        resource=dict(shared_cap_GPU_hours=24,consumed_at_freeze=used,remaining_at_freeze=cap-used,
            allowed_GPUs=[0,2,3],preserve_GPU1=True,main_training_max_GPU_hours=1.25,
            development_max_GPU_hours=.6,conditional_replication_reserve=1.25,
            acceptance_and_trial_reserve=.8,unallocated_reserve=cap-used-3.9,
            on_limit='Watchdog signals owned jobs at23.9 cumulative GPUh; trainer flags SIGTERM and saves model/optimizer/scheduler/RNG at step boundary. Stop all GPU work by24h.')))
    print('Frozen925semantics/1850rows;11 repaired decisions;116 trainable tensors; GPUh remaining',cap-used)


if __name__=='__main__':main()

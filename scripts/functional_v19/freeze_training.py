"""Freeze an input-RGB branch pilot, exact matched positive pool and unchanged gates."""
from functional_v1.common import ROOT,read,write,sha,stamp,GPUJob,EXP as BUDGET
EXP=ROOT/'experiments/functional-v19-20261107';DATA=ROOT/'data/training/functional-v19-20261107'

def main():
    assert not (EXP/'training-protocol.json').exists()
    a=read(DATA/'admission.json');old=read(ROOT/'experiments/functional-v18-20261107/training-protocol.json')
    assert (a['semantic_cases'],a['language_records'],a['source_groups'],a['repeated_positive_presentations_per_epoch'])==(925,1850,240,2002)
    assert sha(DATA/'train-accepted.json')==old['dataset_sha256'] and sha(DATA/'cases.json')==old['cases_sha256']
    assert sha(DATA/'training-pool.json')==old['training_pool_sha256']
    assert sha(EXP/'input-features.json')==a['input_features_sha256']
    assert read(EXP/'bridge-contract-check.json')['zero_output_logits_exact'] is True
    for name,digest in old['code_hashes'].items():assert sha(ROOT/name)==digest,name
    evaluation=read(ROOT/'experiments/functional-v18-20261107/evaluation-protocol.json')
    evaluation.update(at=stamp(),status='round19_input_RGB_bridge_and_unchanged_functional_gates_frozen_before_training',
        comparisons=['Frozen R15e1 parent native16 old4 controls20 controlled75M1.',
                     'Exact archived R18 matched-SFT: same925/1850/2002positive pool,seed20261129,LoRA LR1e-5,2epochs1002steps. R19 adds a16→128→3584 trainable input-RGB feature branch; extra capacity and feature-path change together, not an isolated trained constant-feature ablation.'],
        selection={**evaluation['selection'],'checkpoint':'Epoch1/2 of one photometric_bridge route,seed20261129.'},
        seed_replication='Only if all development gates pass; selected route,seed20261202. Same2epoch1002step schedule. No acceptance before route/checkpoint lock and replicate.',
        initial_function_audit='Fresh3 fixed TRAIN prefixes,zero-output bridge compared with stored R15parent before any update; mismatch stops training.',
        limitations=['Repeated adaptive development, not independent user validation.',
                     'All925semantics/1850language inputs/240groups andpositive targets unchanged;19previous preference positives repeated, no negatives or rewards.',
                     'Bridge uses only fixed16statistics of current input RGB; no target/request/identity fitted feature or corpus standardization.',
                     'Matched SFT lacks extra bridge parameters; a gain would not by itself separate capacity from explicit photometric features.',
                     '24GPUh cap may prevent replication/acceptance; incomplete budget stops are reported, never called validated.'])
    train_groups={c['group_id'] for c in read(DATA/'cases.json')}
    for case_file in evaluation['case_files'].values():
        path=ROOT/case_file['path'];assert sha(path)==case_file['sha256']
        assert not train_groups.intersection(c['group_id'] for c in read(path))
    write(EXP/'evaluation-protocol.json',evaluation)
    code=dict(old['code_hashes']);code.update({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'scripts/functional_v19').glob('*.py')})
    total=GPUJob.consumed(read(BUDGET/'resources.json'));assert 24-total>=2.1,'Not enough reserved budget for the predeclared pilot and conditional followups'
    write(EXP/'training-protocol.json',dict(at=stamp(),status='input_RGB_bridge_pilot_frozen_before_training',
        hypothesis='Seen TRAIN photometric decisions can fit while reused native tasks do not improve. A direct low-dimensional current-RGB path into visual tokens may make magnitude/colour grounding easier; falsified by functional gates, not by CE/L1.',
        routes=['photometric_bridge'],parent_adapter=old['parent_adapter'],parent_sha256=old['parent_sha256'],
        dataset_sha256=a['dataset_sha256'],cases_sha256=a['cases_sha256'],training_pool_sha256=a['pool_sha256'],admission_sha256=sha(DATA/'admission.json'),
        input_features_path=str((EXP/'input-features.json').relative_to(ROOT)),input_features_sha256=sha(EXP/'input-features.json'),
        base_cache_path=old['base_cache_path'],base_cache_sha256=old['base_cache_sha256'],
        source_input_audit_path=old['input_audit_path'],source_input_audit_sha256=old['input_audit_sha256'],
        parent_diagnostic_predictions_path=old['parent_diagnostic_predictions_path'],parent_diagnostic_predictions_sha256=old['parent_diagnostic_predictions_sha256'],
        initial_parity_TRAIN_case_ids=old['initial_parity_TRAIN_case_ids'],
        evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),
        training=dict(rows=2002,unique_language_inputs=1850,unique_semantics=925,photo_groups=240,epochs=2,steps_per_epoch=501,steps=1002,
            learning_rate=1e-5,bridge_learning_rate=1e-3,effective_batch=4,per_device_batch=1,gradient_accumulation=4,
            rank=8,alpha=16,dropout=0,seed=20261129,replicate_seed=20261202,
            anchor_rows_per_epoch=1850,preference_language_records=38,preference_repeat_per_epoch=4,
            objective='Identical assistant-only chosen CE on exact R18 matched-SFT positive presentations. No ranking, negative forward, reward or reference model.',
            image_max_pixels=262144,max_sequence_tokens=2048,trainable_tensors=120,
            trainable_scope='112 languageq/v+4merger LoRA inherited exact, plus4photometric MLP tensors saved in same adapter modules_to_save. Initial last weight/bias zero.',
            quantization='NF4 double,BF16compute,exact original NF4/kbit path; bridge FP32 parameters with autocast during training',
            bridge='16fixed current-RGB features,128GELUhidden,3584output; same vector added to each current-image merged visual token. Original image/literal request/tools unchanged.'),
        resources=dict(cap_GPU_hours=24,occupied_at_freeze=total,allowed_GPUs=[0,2,3],GPU1_original_service_preserved=True,
            shared_watchdog_save_threshold_GPU_hours=23.9,each_training_job_max_seconds=3600,
            interruption='Save120tensor adapter plus optimizer/scheduler/RNG at optimizer boundary; incomplete epoch not eligible; all failed preflights billed.'),
        code_hashes=code,L1_used_for_selection=False,independent_user_feedback=0))
    print('Frozen R19 pilot,120 tensors,2epochs1002steps; exact matched positives; sharedremainingGPUh',24-total)

if __name__=='__main__':main()

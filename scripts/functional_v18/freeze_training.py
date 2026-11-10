"""Freeze actual admitted data and matched functional gates before optimization."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'experiments/functional-v18-20261107'
DATA=ROOT/'data/training/functional-v18-20261107'


def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    assert not (EXP/'training-protocol.json').exists()
    admitted=read(DATA/'admission.json');cache=read(EXP/'training-input-cache-audit.json')
    assert admitted['preference_semantics']==19 and admitted['preference_language_records']==38
    assert admitted['total_presentations_per_epoch']==2002
    assert digest(DATA/'train-accepted.json')==admitted['dataset_sha256']
    assert digest(DATA/'cases.json')==admitted['cases_sha256']
    assert digest(DATA/'preferences.json')==admitted['preferences_sha256']
    assert digest(DATA/'training-pool.json')==admitted['training_pool_sha256']
    old=read(ROOT/'experiments/functional-v15-20261106/training-protocol.json')
    for name,value in old['code_hashes'].items():assert digest(ROOT/name)==value,name
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    evaluation=read(ROOT/'experiments/functional-v15-20261106/evaluation-protocol.json')
    evaluation.update(at=now,status='round18_visual_preferences_matched_routes_and_gates_frozen_before_training',
        comparisons=['Frozen exact R15e1 parent (native16,old4,controls20,controlled75M1) under unchanged editor15/prompt/parser.',
                     'Matched SFT versus CE plus explicit visual counterfactual ranking. Identical parent, positive inputs, record repetitions, seed, optimizer schedule and update count. Rank route adds actual rejected forwards; GPU cost is recorded separately.'],
        assessment='New results require direct anonymous-sheet Codex AI observations and native quality followups. Exact prior input/request/criteria/output pixels may inherit prior final evidence, including all uncertainty and same-render disagreement. Not independent human/user assessment.',
        selection=dict(checkpoint='Epoch1/2 of counterfactual_rank or matched_sft,seed20261129.',
            eligibility='Native>=16/20,old>=4/6,controls20/20,zero native/old major; then controlled>=81/100 and zero major. Stage1 ties with the16/20 parent may proceed to stage2, where81 and zero major require improvement over parent75M1. This changes the comparison baseline, not the numerical functional gates.',
            rank=['More bilingual complete+acceptable native cases','Fewer major effects','Better anonymous paired visual outcome if needed','Earlier checkpoint']),
        seed_replication='Only after every development gate passes; selected route only,seed20261130; lock route/checkpoint before any acceptance inference.',
        initial_function_audit='Fresh3 fixed TRAIN prefixes compared before optimizer in each route; stop before updates if any greedy response differs from recorded reloaded R15 parent. Limited scope, not universal equivalence.',
        limitations=['Repeated adaptive development sets, not a new independent benchmark.',
            'Only19 visually unambiguous synthetic TRAIN preference cases from10 existing photo groups; no new positive labels, photos or requests.',
            'Preference overview sheets hide candidate identity; reviewer had seen numeric diagnostics and knew mappings during native followup. Not fully blind independent label review.',
            'Near-monochrome and ambiguous natural-strength alternatives excluded, with overview/native disagreements retained.',
            'Source isolation/full-pool near-duplicate evidence inherited from unchanged R13/R15 source groups; new group/hash checks are not a fresh full scan.'],
        new_training_started=False,actual_user_feedback=0)
    for row in evaluation['case_files'].values():assert digest(ROOT/row['path'])==row['sha256']
    groups={c['group_id'] for c in read(DATA/'cases.json')}
    for name,row in evaluation['case_files'].items():
        split_groups={c['group_id'] for c in read(ROOT/row['path'])}
        assert not groups.intersection(split_groups),name
    write(EXP/'evaluation-protocol.json',evaluation)
    code=dict(old['code_hashes'])
    code.update({str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'scripts/functional_v18').glob('*.py'))})
    parent=ROOT/'models/from-functional-v11/sft-shadow-correction/seed-01/adapter_model.safetensors'
    assert digest(parent)=='97c84f1799fb319799c86515402574b3b585593763e5388eb30d0df3e0d6ea90'
    write(EXP/'training-protocol.json',dict(at=now,
        status='visual_preferences_matched_optimization_and_functional_gates_frozen_before_training',
        hypothesis='A same-request model often ignores current-image degradation strength. Explicit visually verified own-vs-crossed decisions may improve image conditioning beyond extra positive CE alone. Not assured improvement; functional gates decide.',
        routes=['counterfactual_rank','matched_sft'],parent_adapter=str(parent.parent.relative_to(ROOT)),parent_sha256=digest(parent),
        dataset_sha256=admitted['dataset_sha256'],cases_sha256=admitted['cases_sha256'],admission_sha256=digest(DATA/'admission.json'),
        preferences_sha256=admitted['preferences_sha256'],training_pool_sha256=admitted['training_pool_sha256'],
        input_audit_path=str((EXP/'training-input-cache-audit.json').relative_to(ROOT)),input_audit_sha256=digest(EXP/'training-input-cache-audit.json'),
        target_replay_path=str((DATA/'preferences.json').relative_to(ROOT)),target_replay_sha256=digest(DATA/'preferences.json'),
        base_cache_path=cache['base_cache_path'],base_cache_sha256=cache['base_cache_sha256'],
        negative_cache_path=cache['negative_cache_path'],negative_cache_sha256=cache['negative_cache_sha256'],
        evaluation_protocol_sha256=digest(EXP/'evaluation-protocol.json'),
        parent_diagnostic_predictions_path=str((EXP/'predictions.jsonl').relative_to(ROOT)),parent_diagnostic_predictions_sha256=digest(EXP/'predictions.jsonl'),
        initial_parity_TRAIN_case_ids=['train-020-restore-exposure-strong','train-100-restore-contrast-strong','train-064-restore-saturation-strong'],
        training=dict(rows=2002,unique_language_inputs=1850,unique_semantics=925,photo_groups=240,
            epochs=2,steps_per_epoch=501,steps=1002,learning_rate=1e-5,effective_batch=4,
            per_device_batch=1,gradient_accumulation=4,rank=8,alpha=16,dropout=0,
            seed=20261129,replicate_seed=20261130,ranking_weight=.1,ranking_margin=1.,ranking_score_scale=32.,
            objective='Every row: assistant-only chosen CE. On152 preference presentations/epoch, rank route adds0.1*softplus(1+32*(mean_logp_rejected-mean_logp_chosen)). Sum divided by own token count separately;16/38 candidate pairs have unequal lengths. No reference/reward model. Matched SFT gets same chosen rows/repetitions but no rejected forward.',
            anchor_rows_per_epoch=1850,preference_language_records=38,preference_repeat_per_epoch=4,
            image_max_pixels=262144,max_sequence_tokens=2048,quantization='NF4 double quantization,BF16 compute,exact R13/R15 kbit training path',
            trainable_tensors=116,trainable_scope='112 languageq/v plus4 inherited visual-merger LoRA tensors; one adapter'),
        resources=dict(cap_GPU_hours=24,occupied_at_freeze=read(ROOT/'experiments/functional-v1-20261104/resources.json')['consumed_gpu_hours'],
            allowed_GPUs=[0,2,3],GPU1_original_service_preserved=True,each_training_job_max_seconds=3600,
            shared_watchdog_save_threshold_GPU_hours=23.9,all_training_evaluation_trial_replication_billed=True,
            budget_interruption='Save model,optimizer,scheduler,RNG at optimizer-step boundary. Incomplete checkpoint is not an eligible completed epoch.'),
        source_isolation_inheritance_path='experiments/functional-v15-20261106/source-isolation-inheritance.json',
        source_isolation_inheritance_sha256=digest(ROOT/'experiments/functional-v15-20261106/source-isolation-inheritance.json'),
        code_hashes=code,L1_used_for_selection=False,independent_user_feedback=0))
    print('Frozen matched two-route training:2002 records/epoch,2epochs1002steps; baseline925semantic inputs,19 visual preferences. No training launched by this command.')


if __name__=='__main__':main()

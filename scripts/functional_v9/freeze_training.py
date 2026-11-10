"""Freeze the origin comparison before any round9 GPU model forward."""
import copy
import hashlib
from functional_v1.common import ROOT, read, write, sha, stamp, GPUJob, EXP as BUDGET_EXP
from functional_v7.outcomes import system_prompt, EDITOR_VERSION, SCHEMA_VERSION
from .runtime import EXP, DATA, verify_origin


def main():
    assert not (EXP / 'training-protocol.json').exists()
    assert not (EXP / 'evaluation-protocol.json').exists()
    ledger = read(BUDGET_EXP / 'resources.json')
    assert not any(j['stage'].startswith('v9-') for j in ledger['jobs'])
    consumed = GPUJob.consumed(ledger)
    assert consumed < 12, 'Reserve evaluation and conditional replication within original16GPUh'
    audit = read(EXP / 'origin-audit.json')
    assert audit['status'] == 'CPU_identity_and_shared_input_compatibility_passed'
    assert audit['checked_training_rows'] == 924 and audit['direct_model_forward_calls'] == 0
    for origin in ('base', 'author'):
        verify_origin(origin)
    admission = read(DATA / 'admission.json')
    assert admission['source_groups'] == 160 and admission['photo_domains'] == dict(portrait=80, landscape=80)
    previous_training = read(ROOT / 'experiments/functional-v7-20261105/training-protocol.json')
    for name, digest in previous_training['code_hashes'].items():
        assert sha(ROOT / name) == digest, 'Reused frozen implementation changed'
    for name, key in [('train-accepted.json','dataset_sha256'), ('cases.json','cases_sha256'), ('admission.json','admission_sha256')]:
        assert sha(DATA / name) == previous_training[key]
    input_audit = ROOT / 'experiments/functional-v7-20261105/input-audit.json'
    target_replay = ROOT / 'experiments/functional-v7-20261105/target-replay.json'
    assert sha(input_audit) == previous_training['input_audit_sha256']
    assert sha(target_replay) == previous_training['target_replay_sha256']
    evaluation = copy.deepcopy(read(ROOT / 'experiments/functional-v7-20261105/evaluation-protocol.json'))
    for item in evaluation['case_files'].values():
        assert sha(ROOT / item['path']) == item['sha256']
    evaluation.update(at=stamp(), status='all_round9_cases_metrics_gates_and_comparisons_frozen_before_any_round9_GPU_forward',
        comparisons=['Original Qwen with no adapter under identical new runtime',
                     'Author IEA with no new adapter under identical new runtime; known source overlap, not clean control',
                     'Fresh language-only LoRA on each origin, all four saved epochs',
                     'Frozen strongest same-runtime functional default round7:13/20 native with2 major; old4/6; decision17/20',
                     'Frozen original-pipeline and editor-only diagnostic results retained as context, not newly inferred'],
        selection=dict(eligibility='All stage1 and stage2 gates must pass; exceed strongest same-runtime default and corresponding untrained origin, no major damage. Declined feasible tasks remain failures. No L1, training loss or reward score selects a model.',
            rank=['Most fully completed, acceptable bilingual semantic cases','Fewer major effects','Anonymous paired direct visual comparison','Earlier checkpoint']),
        limitations=['Reuses historically viewed development photos; not pristine independent evaluation', audit['author_overlap_caveat'],
                     'Direct Codex judgments are AI assessment; independent user feedback is zero',
                     'Both routes use the exact original-Qwen shared chat template, not the author original service pipeline'],
        new_training_started=False, actual_user_feedback=0)
    write(EXP / 'evaluation-protocol.json', evaluation)
    rows = admission['actual_rows_each_route']; assert rows == 924
    files = ['scripts/functional_v1/common.py','scripts/functional_v1/editor.py','scripts/functional_v1/runtime.py',
             'scripts/functional_v7/editor_soft.py','scripts/functional_v7/outcomes.py','scripts/functional_v7/runtime.py',
             'scripts/functional_v9/runtime.py','scripts/functional_v9/train.py','scripts/functional_v9/evaluate.py',
             'scripts/functional_v9/freeze_training.py']
    write(EXP / 'training-protocol.json', dict(at=stamp(),
        status='actual_data_runtime_optimizer_and_functional_gates_frozen_before_training',
        hypothesis='Test whether a fresh single language-only adapter avoids inherited enhancement habits. Compare original-Qwen and author-IEA full-weight origins using identical fresh adapter initialization, data, prompts, editor, optimizer and budget. No parent adapter transfer or stacking.',
        routes=['base','author'], origins=audit['origins'], origin_audit_sha256=sha(EXP/'origin-audit.json'),
        origin_input_compatibility='All924 serialized bilingual training examples have identical tokenizer IDs under shared original-Qwen template; eight input-image tensor probes also identical. Asset bytes differ and are not assumed interchangeable.',
        training=dict(rows=rows,epochs=4,steps_per_epoch=231,steps=924,learning_rate=1e-4,effective_batch=4,
            rank=8,alpha=16,dropout=0.,seed=20261115,replicate_seed=20261116,per_device_batch=1,
            gradient_accumulation=4,image_max_pixels=262144,max_sequence_tokens=2048,
            quantization='NF4, double quantization, BF16 compute',optimizer='adamw_torch',weight_decay=0.,
            scheduler='cosine',warmup_ratio=.05,max_grad_norm=1.,save='every complete epoch',
            assistant_only_decision_JSON_supervision=True,trainable_tensors=112,visual_parameters_frozen=True,
            fresh_adapter_random_seed_reset_before_creation=True),
        dataset_sha256=sha(DATA/'train-accepted.json'),cases_sha256=sha(DATA/'cases.json'),admission_sha256=sha(DATA/'admission.json'),
        input_audit_path=str(input_audit.relative_to(ROOT)),input_audit_sha256=sha(input_audit),
        target_replay_path=str(target_replay.relative_to(ROOT)),target_replay_sha256=sha(target_replay),
        evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),code_hashes={name:sha(ROOT/name) for name in files},
        editor=EDITOR_VERSION,schema=SCHEMA_VERSION,runtime='functional_v9.runtime',
        system_prompt_sha256=hashlib.sha256(system_prompt().encode()).hexdigest(),source_counts=admission,
        source_overlap_caveat=audit['author_overlap_caveat'],
        resource=dict(shared_cap_GPU_hours=16,consumed_before_round9=consumed,remaining_at_freeze=16-consumed,
            allowed_GPUs=[0,2,3],preserve_GPU1=True,estimated_two_training_GPU_hours=1.6,
            estimated_stage1_evaluation_GPU_hours=.7,conditional_replication_and_trial_reserve_GPU_hours=1.5,
            on_limit='Watchdog stops owned experiment jobs before cap; save state and report, never raise budget or continue unbounded.')))
    print('Round9 frozen before GPU forwards:924 unchanged bilingual rows,160 photos80/80; two fresh origins; remaining GPUh',16-consumed)

if __name__ == '__main__':
    main()

"""Freeze functional gates separately from later data admission/training-code lock."""
from functional_v1.common import ROOT, read, write, sha, stamp
from .source_bank import EXP, DATA


def main():
    out = EXP/'evaluation-protocol.json'
    assert not out.exists(), 'Never move the gate after seeing candidate outcomes'
    controlled = ROOT/'data/training/functional-v2-20261104'
    files = {
        'native_development': DATA/'development-native-cases.json',
        'native_acceptance': DATA/'acceptance-native-cases.json',
        'controlled_development': controlled/'development-final-cases.json',
        'controlled_acceptance': controlled/'acceptance-final-cases.json',
        'old_native_regressions': controlled/'natural-use-final-cases.json',
    }
    for stem in ('development', 'acceptance'):
        assert sha(files[f'native_{stem}']) == read(DATA/f'{stem}-native-freeze.json')['cases_sha256']
    write(out, dict(at=stamp(), status='evaluation_gates_frozen_before_round6_training_or_developer_model_outputs',
        case_files={k: dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for k, p in files.items()},
        unit='Original semantic case counts only; ZH and EN both must complete with acceptable quality. Translation rows are not independent samples. Report source-group overlaps separately.',
        rating=dict(completion=['completed', 'partial', 'failed', 'uncertain'], quality=['acceptable', 'minor_issue', 'major_issue', 'uncertain'],
            qualified='Both language outputs completed and quality acceptable; all other combinations unqualified. Major side effect counted once per semantic case if either language has one.'),
        assessment='AI assisted: actual executed images directly inspected by Codex, anonymous panels before scoring; context may identify earlier results so this is not double-blind human review. Frozen original7B reviewer previously failed functional calibration and remains fallible/reference-only. Keep disagreements/uncertainty visible; never call a score user approval.',
        stage1=dict(cases='native_development20 plus old_native_regressions6, each bilingual',
            minimum_native_development_qualified=16, minimum_old6_qualified=4, maximum_major_cases=0,
            stop='A checkpoint that fails any of these three requirements cannot qualify. Its100 controlled predictions/review may be skipped with that omission explicit; never fabricate an aggregate controlled score.'),
        stage2=dict(cases='controlled_development100 bilingual', minimum_qualified=81, maximum_major_cases=0,
            comparisons='Same exact v1 prompt, editor/parser and raw image across rawQwen, existing default, round2 parent and both new routes. Report editor/runtime identity. No measured-input note or fifth-round head.'),
        selection=dict(eligibility='All stage1 and stage2 gates, visible improvement over existing default on functional completion and no new major damage.',
            rank=['Most jointly completed, acceptable cases across developer sets; disclose reused-photo overlaps',
                  'Fewer major effects (all qualified must have zero)', 'Blind paired direct review of tied checkpoints', 'Earlier checkpoint'],
            l1_or_numeric_parameter_error_used=False),
        acceptance=dict(open_after='Select route/checkpoint on development, lock actual adapter/parser/prompt/editor, then seed replicate selected route only if developmental gates pass. Final acceptance outcomes never choose checkpoint or parameters.',
            native_cases=20, controlled_cases=100, qualitative_gate='Native>=16/20 bilingual complete acceptable, zero major; controlled>=81/100 complete acceptable, zero major; report every task/domain/failure and compare current under identical inputs. Passing is AI evidence, not proof of high practical usefulness or user approval.',
            no_posthoc_checkpoint_changes=True, no_parameter_tuning_on_acceptance=True),
        candidate_training_intention=dict(routes=['RawQwen + newLoRA', 'Round2curriculum-e3 + continuedLoRA'],
            same_final_admitted_dataset_prompt_seed_updates=True, quantization='NF4 QLoRA', target_modules='language q/v only',
            rank=8, alpha=16, effective_batch=4, learning_rate=1e-4, epochs=4, seed=20261110, replicate_seed=20261111,
            save='Every complete epoch',
            admission_gate='No training starts before100 native targets have executed-image visual admission, bilingual equivalence checked, and final dataset/target replay, isolation, actual training/runtime code hashes and optimizer settings locked in a separate training protocol. If cohort changes before training, preserve an amendment and actual counts.'),
        resource=dict(shared_original_gpu_hour_cap=16, allowed_gpu_indices=[0, 2, 3], gpu1_original_service_preserved=True,
            ledger='experiments/functional-v1-20261104/resources.json', includes='All occupied model load, label proposals/screening, training, evaluation, replication and trial time'),
        independent_user_feedback=0, new_training_started=False,
        implementation_sha256=sha(__file__)))
    print('Functional evaluation gate frozen; target admission/training lock still required')


if __name__ == '__main__': main()

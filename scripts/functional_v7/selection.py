"""Apply frozen gates and register both final-epoch diagnostics honestly."""
import json
import shutil

from functional_v1.common import ROOT, read, write, sha, stamp, GPUJob
from .visual_review import EXP, EVAL


def main():
    destination = EXP / 'development-selection.json'
    assert not destination.exists()
    baselines = ['current-default-new-runtime', 'round6-parent-new-runtime']
    candidates = [f'{r}-e{e}' for e in (1, 2, 3, 4) for r in ('language', 'language-merger')]
    summaries = {s: {m: read(EVAL / s / m / 'direct-functional-summary.json')
                     for m in baselines + candidates} for s in ('native-development', 'old-native')}
    decisions = {m: read(EVAL / 'decision-development' / m / 'decision-summary.json') for m in baselines + candidates}
    assert all(x['status'] == 'completed' and not x['missing_unique_images']
               for group in summaries.values() for x in group.values())
    protocol = read(EXP / 'training-protocol.json')
    evaluation = read(EXP / 'evaluation-protocol.json')
    gates = evaluation['stage1']
    eligible = []
    table = {}
    for model in baselines + candidates:
        n, o, d = summaries['native-development'][model], summaries['old-native'][model], decisions[model]
        passed = (n['fully_completed_both_languages'] >= gates['native_development_minimum']
                  and o['fully_completed_both_languages'] >= gates['old_native_minimum']
                  and n['major_quality_issues'] == o['major_quality_issues'] == gates['maximum_major_cases']
                  and d['passed_both_languages'] == gates['decision_controls_required'])
        table[model] = dict(native_completed=n['fully_completed_both_languages'], native_major=n['major_quality_issues'],
                            old_completed=o['fully_completed_both_languages'], old_major=o['major_quality_issues'],
                            decision_controls=d['passed_both_languages'], stage1_passed=passed,
                            feasible_native_refusals=n['declined_semantic_cases'],
                            feasible_old_refusals=o['declined_semantic_cases'])
        if model in candidates and passed:
            eligible.append(model)
    assert not eligible, 'A qualified checkpoint needs stage2 and conditional replication, not failed-pilot registration'
    routes = ('language', 'language_merger')
    runs = {r: read(EXP / 'runs' / f'{r}-seed-20261112' / 'status.json') for r in routes}
    audits = [read(EXP / 'runs' / f'{r}-seed-20261112' / 'token-audit.json') for r in routes]
    assert audits[0] == audits[1] and len(audits[0]) == 924
    for route, run in runs.items():
        assert run['status'] == 'completed' and run['step'] == 924 and run['epoch'] == 4
        assert run['changed_language_tensors'] == 112
        assert run['changed_visual_merger_tensors'] == (4 if route == 'language_merger' else 0)
    admission = read(ROOT / 'data/training/functional-v7-20261105/admission.json')
    result = dict(at=stamp(), status='failed_practical_gate', qualified=False, promoted=False,
                  selected_model_id=None, trial_default_changed=False, stage1_qualified_checkpoints=[],
                  controlled_development_status='not_run_all_candidates_failed_predeclared_stage1',
                  seed_replication_started=False, acceptance_status='closed_not_evaluated', results=table,
                  training_protocol_sha256=sha(EXP / 'training-protocol.json'),
                  evaluation_protocol_sha256=sha(EXP / 'evaluation-protocol.json'),
                  same_runtime_superiority_clarification_sha256=sha(EXP / 'selection-eligibility-clarification.json'),
                  explanation='All trained checkpoints fail native>=16/20 and zero-major gates. New decision controls pass20/20; editing remains14/20 at best with2-3 major cases. Refusals on feasible tasks remain uncompleted. No L1-based selection.',
                  diagnostic_choice='Each route final epoch4 is exposed as an end-of-training diagnostic, not a selected winner. All eight checkpoint results remain visible.',
                  training_runs=runs, admission=admission, models={}, independent_user_feedback=0)
    for route in routes:
        evaluation_id = route.replace('_', '-') + '-e4'
        checkpoint = EXP / 'runs' / f'{route}-seed-20261112' / 'checkpoint-924'
        assert sha(checkpoint / 'adapter_model.safetensors') == runs[route]['adapter_sha256']
        folder = ROOT / 'models/from-functional-v6' / ('sft-outcome-' + route.replace('_', '-')) / 'seed-01'
        folder.mkdir(parents=True, exist_ok=False)
        for filename in ('adapter_config.json', 'adapter_model.safetensors'):
            shutil.copy2(checkpoint / filename, folder / filename)
        model_id = 'functional-v7-' + route.replace('_', '-') + '-20261112'
        item = dict(model_id=model_id, evaluation_id=evaluation_id,
                    display_name='第七轮 · ' + ('语言＋视觉连接层' if route == 'language_merger' else '语言层') + ' · 第4轮（未通过）',
                    trial_status='实验模型，AI辅助评审；本轮未过功能门槛。逆光人脸、夜景噪声、高光与力度仍会失败；拒绝行为也不保证识别全部风险。',
                    adapter=str(folder.relative_to(ROOT)), adapter_sha256=sha(folder / 'adapter_model.safetensors'),
                    parent=protocol['parent'], initial_adapter_sha256=protocol['parent_sha256'],
                    checkpoint=str(checkpoint.relative_to(ROOT)), epoch=4, step=924,
                    runtime_family='functional_v7', training_route=route,
                    editor=admission['editor'], schema=admission['schema'], dataset_sha256=admission['dataset_sha256'],
                    cases_sha256=admission['cases_sha256'], semantic_cases=462, training_records=924,
                    original_photo_groups=160, photo_domains=admission['photo_domains'],
                    outcome_counts=admission['outcome_counts'],
                    original_GIER_English_cases=97, AI_authored_cases=365,
                    status='diagnostic_trial_only_not_promoted', diagnostic_choice=result['diagnostic_choice'],
                    load_method='Original Qwen base plus this ONE adapter; never stack historical adapters.',
                    native_development_summary=summaries['native-development'][evaluation_id],
                    old_native_summary=summaries['old-native'][evaluation_id],
                    decision_development_summary=decisions[evaluation_id], user_approval=False, at=stamp())
        write(folder / 'metadata.json', item)
        result['models'][model_id] = item
    ledger = read(ROOT / 'experiments/functional-v1-20261104/resources.json')
    result['resource_snapshot'] = dict(at=stamp(), occupied_gpu_hours=GPUJob.consumed(ledger), limit_gpu_hours=16)
    write(destination, result)
    write(EXP / 'training-verification.json', dict(at=stamp(), status='passed_training_mechanics_only',
          matched_token_audits=True, unique_language_records_each=924, optimizer_steps_each=924,
          checkpoint_steps=[231, 462, 693, 924], language_adapter_tensors_each=112,
          visual_merger_tensors_added=4, parent_transfer_exact=True,
          training_uses_assistant_JSON_only=True, finite_gradients=True,
          final_weights_changed=True, functional_usefulness_established=False))
    print(json.dumps(dict(status=result['status'], results=table, models=list(result['models']), resources=result['resource_snapshot'])))


if __name__ == '__main__':
    main()

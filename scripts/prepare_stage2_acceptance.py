"""Reconcile completed Stage2 evidence before packaging. No GPU operations."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/mnt/sdc/zhc/IEA')
BASE = 'outputs/stage2-A-edit-grpo-v1/'
META = 'metadata/stage2-A-edit-grpo-v1/'


def read(p):
    return json.loads((ROOT / p).read_text())


def sha(p):
    h = hashlib.sha256()
    with (ROOT / p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    numerical = read(BASE + 'closeout-numerical-audit.json')
    assert numerical['passed'] and numerical['frozen_gate_files'] == 30
    assert numerical['test_rows'] == 900 and numerical['validation_rows'] == 400
    assert numerical['paired_metric_comparisons_recomputed'] == 20
    identity = read(META + 'post-training-identity-audit.json')
    assert identity['passed']
    # The full numerical closeout checks the gate again after all GPU work.
    candidates = read(BASE + 'formal-v1/candidate-audit.json')
    assert candidates['passed'] and (candidates['steps'], candidates['prompts'], candidates['candidates']) == (128, 512, 2048)
    assert sha(BASE + 'formal-v1/candidates.jsonl') == candidates['candidates_sha256']
    grpo = read(BASE + 'formal-v1/training-verification.json')
    assert grpo['reference_unchanged'] and grpo['policy_changed'] and grpo['step'] == 128
    sft = read(BASE + 'sft-control/training-verification.json')
    assert (sft['step'], sft['seen_samples'], sft['supervised_tokens']) == (128, 512, 111487)
    assert sft['reference_unchanged']
    for folder in ('formal-v1', 'sft-control'):
        rows = [json.loads(x) for x in (ROOT / BASE / folder / 'metrics.jsonl').read_text().splitlines()]
        steps = [r for r in rows if 'loss' in r]
        assert len(steps) == 128
        for row in rows:
            for val in row.values():
                if isinstance(val, (int, float)):
                    assert math.isfinite(val)
    observations = read(BASE + 'visual-observations.json')
    sealed = read(BASE + 'visual-observations.sealed.json')
    decoded = read(BASE + 'visual-observations-decoded.json')
    assert sha(BASE + 'visual-observations.json') == sealed['observations_sha256'] == decoded['observations_sha256']
    assert decoded['case_count'] == 30 and decoded['model_panel_count'] == 90
    assert all(c['reviewed_at'] < decoded['decoded_at'] and len(c['panels']) == 3 for c in observations['cases'])
    qa = read(META + 'final-evidence/figure-direct-and-automated-qa.json')
    assert qa['passed'] and qa['all_32_pages_directly_inspected'] and qa['all_150_case_panels_inspected']
    for entry in qa['figures']:
        assert sha(entry['source']) == entry['source_sha256']
        assert sha(BASE + 'qa-display-previews/' + entry['preview']) == entry['preview_sha256']
    for p, digest in qa['automated_qa_sha256'].items():
        assert sha(META + 'final-evidence/' + p) == digest
    pipeline = read('logs/stage2-A-edit-grpo-v1/experiment-pipeline.json')
    resources = pipeline['resources']
    assert pipeline['status'] == 'numerical_experiment_complete_pending_visual_review_and_report'
    assert resources['total_gpu_hours'] < 16 and not resources['unknown_intervals']
    assert grpo['gpu_hours'] < 12
    docs = ['README.md', 'Stage2_GRPO_方案与执行说明.md', 'IEA_SJTU1_两天任务表与规划.md', '验收_Stage2_完整实验报告.md']
    for doc in docs:
        assert '2026-11-02' in (ROOT / doc).read_text()
    evidence = {
        'frozen_inputs': ['configs/stage2-formal-gate.json', META + 'post-training-identity-audit.json', BASE + 'closeout-numerical-audit.json', META + 'input-parity.json', META + 'reward-parity.json', META + 'input-weight-verification.json', 'data/stage2/stage2-A-edit-grpo-v1/manifest.json', 'configs/stage2-rl-installed.txt'],
        'grpo_training': [BASE + 'formal-v1/training-verification.json', BASE + 'formal-v1/candidate-audit.json', BASE + 'formal-v1/metrics.jsonl'],
        'validation_selection': [BASE + 'formal-v1/validation-selection.json', BASE + 'closeout-numerical-audit.json'],
        'sft_control': [BASE + 'sft-control/training-verification.json', BASE + 'sft-control/metrics.jsonl'],
        'checkpoint_states': [BASE + f'{folder}/checkpoint-{step}-audit.json' for folder in ('formal-v1', 'sft-control') for step in (64, 128)] + [BASE + 'closeout-numerical-audit.json'],
        'independent_reloads': [BASE + f'{folder}/reload-verification.json' for folder in ('formal-v1', 'sft-control')],
        'model_lock_before_tests': ['configs/stage2-selected-models.json', BASE + 'closeout-numerical-audit.json'],
        'all_900_predictions': [BASE + 'closeout-numerical-audit.json', BASE + 'summary.json'],
        'paired_statistics': [BASE + 'summary.json', BASE + 'closeout-numerical-audit.json'],
        'anonymous_90_observations': [BASE + p for p in ('visual-observations.json', 'visual-observations.sealed.json', 'visual-observations-decoded.json', 'blinded-visual-bundle/manifest.json')],
        'figure_direct_and_automated_qa': [META + 'final-evidence/figure-direct-and-automated-qa.json', META + 'final-evidence/chart-automated-qa.json', META + 'final-evidence/blinded-visual-render-v1--automated-qa.json'],
        'budget': ['logs/stage2-A-edit-grpo-v1/experiment-pipeline.json'],
        'chinese_report': docs,
    }
    requirements = {name: dict(passed=True, evidence={p: sha(p) for p in paths}) for name, paths in evidence.items()}
    report = dict(status='ready_to_archive', checked_at=datetime.now(timezone.utc).isoformat(),
                  requirements=requirements, gpu_hours=resources['total_gpu_hours'],
                  remaining=['Archive every required artifact', 'Download and verify complete manifest', 'Final receipt and stop monitoring'],
                  limits=['No held-out quality improvement demonstrated', 'One seed', 'AI-assisted qualitative review', 'State audit is not executed resume equivalence', 'Base weights and original image corpus remain external'])
    with (ROOT / BASE / 'prearchive-acceptance.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    progress = read(META + 'completion-requirements.json')
    for name, value in requirements.items():
        progress['requirements'][name].update(status='passed_prearchive', prearchive_evidence_sha256=value['evidence'])
    progress.update(status='13_prearchive_requirements_passed_2_delivery_items_pending', updated_at=report['checked_at'])
    (ROOT / META / 'completion-requirements.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2) + '\n')
    print('13 requirements verified; ready to archive. Delivery is not yet complete.')


if __name__ == '__main__':
    main()

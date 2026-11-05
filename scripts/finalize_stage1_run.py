"""Summarize a completed, reloaded run without performing training or test inference."""
import csv
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess

from supervise_stage1_training import ROOT, RUN, OUT, stamp, write_json


def read(path):
    return json.loads(path.read_text())


def main():
    global RUN, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=['A', 'B'], default='A')
    cli = parser.parse_args()
    b_manifest = None
    if cli.experiment == 'B':
        RUN = ROOT / 'logs/stage1-B-3k'
        OUT = ROOT / 'outputs/stage1-B-image-edit-3k-v1'
        subprocess.run([str(ROOT / 'envs/iea/bin/python'), str(ROOT / 'scripts/verify_stage1_B.py')], check=True)
        b_manifest = read(ROOT / 'configs/stage1-B-3k-manifest.json')
    subprocess.run([str(ROOT / 'scripts/train_stage1_frozen.sh'), '--check-only'], check=True)
    status = read(RUN / 'status.json')
    state = read(OUT / 'trainer_state.json')
    metrics = read(OUT / 'train_results.json')
    evaluation = read(OUT / 'eval_results.json')
    artifact = read(RUN / 'artifact-verification.json')
    reload_report = read(RUN / 'reload-verification.json')
    decision = read(ROOT / 'configs/stage1-A-3k-decision.json')
    if b_manifest:
        decision = {**decision, 'formal_config_sha256': b_manifest['config_sha256']}
    assert status['status'] == 'completed' and not status['process_alive']
    assert state['global_step'] == 376 and state['epoch'] == 2.0
    assert all(math.isfinite(row[key]) for row in state['log_history']
               for key in ('loss', 'grad_norm', 'eval_loss') if key in row)
    assert artifact['final_equals_best_checkpoint'] and reload_report['adapter_reload_verified']
    assert reload_report['training_updates_during_reload'] == 0
    assert not reload_report['test_set_evaluated']
    digest = hashlib.sha256((OUT / 'adapter_model.safetensors').read_bytes()).hexdigest()
    assert digest == artifact['adapter_sha256'] == reload_report['adapter_sha256']
    assert artifact['best_checkpoint'] == reload_report['best_checkpoint'] == state['best_model_checkpoint']
    assert math.isclose(evaluation['eval_loss'], state['best_metric'], rel_tol=1e-6, abs_tol=1e-7)
    validation = [row for row in state['log_history'] if 'eval_loss' in row]
    assert [row['step'] for row in validation] == [94, 188, 282, 376]
    with (RUN / 'gpu.csv').open() as stream:
        gpu = list(csv.DictReader(stream))
    records = [json.loads(line) for line in (OUT / 'trainer_log.jsonl').read_text().splitlines()]
    wall = (datetime.fromisoformat(status['finished_at']) - datetime.fromisoformat(status['started_at'])).total_seconds()
    eval_seconds = sum(row['eval_runtime'] for row in validation)
    adjusted_seconds = metrics['train_runtime'] - eval_seconds
    summary = dict(
        status='completed_and_reload_verified', experiment=cli.experiment, report_created_at=stamp(),
        started_at=status['started_at'], completion_confirmed_at=status['finished_at'],
        reload_verified_at=reload_report['verified_at'],
        gpu_index=2, global_step=376, epochs=2.0,
        train_samples=3000, validation_samples=200, held_out_test_samples=200,
        training_sample_exposures=6000,
        exposure_count_basis='3000 samples x 2 complete epochs, no dropped tail; inferred from the frozen loader and final epoch/step state, not a separate sample counter',
        base_model=str(ROOT / 'models/Qwen2.5-VL-7B-Instruct'),
        base_model_revision=decision['model_revision'], repository_commit=decision['repository_commit'],
        formal_config_sha256=decision['formal_config_sha256'],
        split_manifest_sha256=decision['split_manifest_sha256'],
        adapter_path=str(OUT), best_checkpoint=state['best_model_checkpoint'], adapter_sha256=digest,
        best_validation_loss=state['best_metric'], validation_history=validation,
        train_metrics=metrics, final_validation_metrics=evaluation,
        runtime=dict(launch_to_completion_confirmation_seconds=wall,
                     trainer_loop_including_periodic_validation_and_saving_seconds=metrics['train_runtime'],
                     periodic_validation_seconds=eval_seconds,
                     trainer_loop_minus_periodic_validation_seconds=adjusted_seconds,
                     samples_per_second_including_periodic_validation=6000 / metrics['train_runtime'],
                     approximate_samples_per_second_excluding_periodic_validation=6000 / adjusted_seconds,
                     note='The subtracted estimate still includes checkpoint/other Trainer overhead; wall time includes loading, preprocessing, a final best-model validation and up to a polling interval before completion confirmation. Trainer sample throughput uses 376 x 16 = 6016, while two tail batches reduce actual exposures to 6000.'),
        memory=dict(sampled_device_peak_mib=max(int(row['memory_used_mib']) for row in gpu),
                    sampled_device_peak_gib=max(int(row['memory_used_mib']) for row in gpu) / 1024,
                    pytorch_peak_allocated_gib=max(row.get('vram_allocated', 0) for row in records),
                    pytorch_peak_reserved_gib=max(row.get('vram_reserved', 0) for row in records),
                    sampled_peak_temperature_c=max(int(row['temperature_c']) for row in gpu),
                    device_sampling_interval_seconds=10,
                    note='Device sampling can miss instantaneous peaks; PyTorch values are rounded to 2 decimal places by LLaMA-Factory.'),
        all_logged_losses_and_gradient_norms_finite=True, reported_training_alerts=status['alerts'],
        final_equals_best_checkpoint=True, reloaded_adapter_tensors_equal=reload_report['adapter_tensors_exactly_equal'],
        reload_probe_loss=reload_report['probe']['loss'], test_set_evaluated=False,
        exit_status_available=status['exit_status_available'],
        limitations=['Validation loss and a training-probe reload do not establish image-editing quality or improvement over Base.'])
    if not status['exit_status_available']:
        summary['limitations'].append('The original supervisor failed before writing its first status and was reattached to the unchanged trainer; the original exit code is unavailable. Completion is verified from process termination, final state, results and adapter checks.')
    write_json(RUN / 'summary.json', summary)
    if b_manifest:
        summary['sampling_manifest'] = str(ROOT / 'configs/stage1-B-3k-manifest.json')
        summary['unique_source_rows'] = b_manifest['unique_source_rows']
        summary['unique_source_groups'] = b_manifest['unique_source_groups']
        summary['audited_token_exposures'] = b_manifest['B_token_exposures']
        summary['limitations'].extend(b_manifest['limitations'])
        write_json(RUN / 'summary.json', summary)
    status.update(artifact_verification=artifact, reload_verification_report=str(RUN / 'reload-verification.json'),
                  completion_summary=str(RUN / 'summary.json'), full_model_reload_verified=True,
                  acceptance_finished_at=reload_report['verified_at'], updated_at=stamp())
    write_json(RUN / 'status.json', status)
    assets = read(ROOT / 'configs/stage1-assets.json')
    assets['status'] = 'A_and_B_completed_and_reload_verified' if b_manifest else 'formal_training_completed_and_reload_verified'
    assets['next_steps'] = [
        'Finish validation generation diagnostics and lock all selected model identities before any test inference.',
        'Run the frozen 200-row evaluation for Base, A, B, Official and supplementary BaseSchema; retain all failures.',
        'Complete paired analysis, blinded visual review, result archiving and final acceptance report.'
    ]
    experiment_key = 'control_experiment' if b_manifest else 'current_experiment'
    assets.setdefault(experiment_key, {}).update(selected_adapter=str(OUT), best_checkpoint=state['best_model_checkpoint'],
                                        adapter_sha256=digest, completion_summary=str(RUN / 'summary.json'),
                                        test_set_evaluated=False)
    write_json(ROOT / 'configs/stage1-assets.json', assets)
    print('FORMAL_RUN_FINALIZED', json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

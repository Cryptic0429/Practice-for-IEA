"""Verify completed experimental evidence; negative quality results can still pass.

Run only after all outputs, direct visual observations and the report exist.
Archive-byte verification is a separate subsequent step, avoiding a self-hash.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from visual_review_records import validate

ROOT = Path('/mnt/sdc/zhc/IEA')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    evaluation = ROOT / 'outputs/evaluation-v1'
    protocol_path = ROOT / 'configs/evaluation-protocol-v1.json'
    protocol = read(protocol_path)
    lock_path = ROOT / 'configs/evaluation-models-v1.json'
    lock = read(lock_path)
    requirements = []

    def record(name, evidence):
        requirements.append(dict(name=name, passed=True, evidence=evidence))

    assert protocol['status'] == 'frozen' and lock['all_selected_models_verified']
    assert sha(protocol_path) == lock['protocol_sha256']
    for path, digest in protocol['code_hashes'].items():
        assert sha(Path(path)) == digest, path
    for model, entry in lock['models'].items():
        for item in entry['identity_files']:
            assert sha(Path(item['path'])) == item['sha256'], item['path']
        for item in entry['weight_shards']:
            stat = Path(item['path']).stat()
            assert (stat.st_size, stat.st_mtime_ns) == (item['bytes'], item['mtime_ns'])
    record('Frozen protocol, code and model identities unchanged',
           dict(protocol_sha256=sha(protocol_path), model_lock_sha256=sha(lock_path),
                weight_note='Full shard SHA evidence is bound in the model lock; size/mtime rechecked here.'))

    split_folder = Path(protocol['split_directory'])
    manifest = read(split_folder / 'manifest.json')
    datasets = {}
    for split, count in [('train', 3000), ('validation', 200), ('test', 200)]:
        path = split_folder / f'{split}.json'
        assert sha(path) == manifest['files'][split]['sha256']
        datasets[split] = read(path)
        assert len(datasets[split]) == count
    groups = {name:{row['group_id'] for row in rows} for name,rows in datasets.items()}
    assert not (groups['train'] & groups['validation'] or groups['train'] & groups['test']
                or groups['validation'] & groups['test'])
    assert len(groups['test']) == len(groups['validation']) == 200
    record('Frozen splits and source-group separation',
           dict(rows={s:len(rows) for s,rows in datasets.items()}, groups={s:len(v) for s,v in groups.items()},
                limitation='This checks the frozen exact/path/URL grouping; no near-duplicate certification.'))
    sampling = read(ROOT / 'configs/stage1-B-3k-manifest.json')
    assert sha(Path(sampling['train_path'])) == sampling['train_sha256']
    assert sha(Path(sampling['source_train_path'])) == sampling['source_train_sha256']
    assert sha(split_folder / 'manifest.json') == sampling['split_manifest_sha256']
    assert sha(Path(sampling['config_path'])) == sampling['config_sha256']
    sampled = read(Path(sampling['train_path']))
    assert len(sampled) == sampling['train_rows'] == 3000
    assert len({row['group_id'] for row in sampled}) == sampling['unique_source_groups'] == 901
    assert {row['group_id'] for row in sampled} <= groups['train']
    record('B sampling input and original training pool unchanged',
           dict(manifest_sha256=sha(ROOT / 'configs/stage1-B-3k-manifest.json'),
                train_sha256=sampling['train_sha256']))

    for model in ['A', 'B']:
        folder = ROOT / f'logs/stage1-{model}-3k'
        summary = read(folder / 'summary.json')
        artifact = read(folder / 'artifact-verification.json')
        reload = read(folder / 'reload-verification.json')
        output = Path(summary['adapter_path'])
        trainer = read(output / 'trainer_state.json')
        assert summary['status'] == 'completed_and_reload_verified'
        assert trainer['global_step'] == summary['global_step'] == artifact['global_step'] == 376
        assert trainer['best_model_checkpoint'] == summary['best_checkpoint'] == artifact['best_checkpoint']
        assert math.isfinite(trainer['best_metric']) and trainer['best_metric'] == summary['best_validation_loss']
        assert [r['step'] for r in summary['validation_history']] == [94, 188, 282, 376]
        assert summary['all_logged_losses_and_gradient_norms_finite']
        assert artifact['final_equals_best_checkpoint'] and artifact['adapter_tensor_count'] == 112
        assert reload['adapter_reload_verified'] and reload['adapter_tensors_exactly_equal'] == 112
        assert reload['training_updates_during_reload'] == 0 and not reload['test_set_evaluated']
        assert reload['probe_samples'] == 1 and math.isfinite(reload['probe']['loss'])
        assert sha(output / 'adapter_model.safetensors') == summary['adapter_sha256'] == artifact['adapter_sha256'] == reload['adapter_sha256']
        assert sha(ROOT / f'configs/qlora-stage1-{model}-3k.yaml') == summary['formal_config_sha256']
        for name in ['optimizer.pt', 'scheduler.pt', 'adapter_model.safetensors', 'trainer_state.json']:
            assert (Path(summary['best_checkpoint']) / name).is_file()
        record(f'{model}: training, validation selection, checkpoint and zero-update reload',
               dict(summary_sha256=sha(folder / 'summary.json'), adapter_sha256=summary['adapter_sha256']))

    expected = [row['id'] for row in datasets['test']]
    parity = read(evaluation / 'input-tensor-parity-test-audit.json')
    assert parity['sample_count'] == 200 and parity['source_split'] == 'test'
    assert parity['all_probes_identical'] and [row['id'] for row in parity['probes']] == expected
    assert not parity['loaded_model_weights'] and parity['model_forward_calls'] == 0
    assert parity['protocol_sha256'] == lock['protocol_sha256']
    assert all(protocol['models'][model]['model_name_or_path'] == protocol['models']['Base']['model_name_or_path']
               for model in ['A', 'B', 'BaseSchema'])
    record('All 200 primary-condition input reconstructions match',
           dict(audit_sha256=sha(evaluation / 'input-tensor-parity-test-audit.json'),
                scope='CPU reconstruction of Base/Official preprocessing; A/B use the same base processor. BaseSchema intentionally changes the prompt. No new forward calls.'))
    summary = read(evaluation / 'summary.json')
    assert summary['sample_count'] == summary['independent_source_groups'] == 200
    assert set(summary['models']) == set(lock['models'])
    for model in lock['models']:
        folder = evaluation / f'test-{model}-full'
        status, run = read(folder / 'status.json'), read(folder / 'run.json')
        assert status['status'] == 'completed' and status['completed'] == status['total'] == 200
        assert datetime.fromisoformat(run['started_at']) >= datetime.fromisoformat(lock['frozen_at'])
        assert run['identity']['protocol_sha256'] == lock['protocol_sha256']
        rows = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
        assert [row['id'] for row in rows] == expected and summary['models'][model]['n'] == 200
        for row in rows:
            score = row['scoring']
            assert score['input_readable'] and score['reference_readable'] and not score['metric_error']
            assert all(math.isfinite(v) for v in score['edited_metrics'].values())
            assert Path(score['edited_image_path']).is_file()
            if not score['editor_success']:
                assert score['fallback_used'] and score['baseline_metrics'] == score['edited_metrics']
        record(f'{model}: all 200 frozen test IDs and failures retained',
               dict(predictions_sha256=sha(folder / 'predictions.jsonl'),
                    executable_count=summary['models'][model]['editor_success_count']))
    assert read(ROOT / 'logs/evaluation-v1/queue-status.json')['status'] == 'completed'
    assert not read(evaluation / 'validation-diagnostics.json')['pending_models']
    record('Five-condition metrics and predefined paired analyses complete', sha(evaluation / 'summary.json'))

    ids = read(Path(protocol['visual_ids_path']))
    assert sha(Path(protocol['visual_ids_path'])) == protocol['visual_ids_sha256']
    observations = read(evaluation / 'visual-review-observations.json')
    validate(observations, ids)
    assert read(evaluation / 'visual-review-decoded.json')['observations_sha256'] == sha(evaluation / 'visual-review-observations.json')
    audit = read(evaluation / 'figure-audit-summary.json')
    assert audit['automated_checks_passed'] and audit['expected_pdf_count'] == len(audit['figures']) == 32
    manual = read(evaluation / 'final-visual-qa.json')
    assert manual['direct_visual_inspection_completed'] and len(manual['figures']) == 32
    assert {item['path'] for item in manual['figures']} == {item['path'] for item in audit['figures']}
    for item in manual['figures']:
        assert item['passed'] and item['note'].strip()
        assert sha(Path(item['path']).with_suffix('.png')) == item['png_sha256']
    record('30 preselected cases, 150 concealed-label observations and 32 inspected figures',
           dict(observations_sha256=sha(evaluation / 'visual-review-observations.json'),
                visual_qa_sha256=sha(evaluation / 'final-visual-qa.json'),
                scope='AI-assisted qualitative inspection, not a human user study.'))
    for name in ['验收_T4-T7_完整实验报告.md', '验收_T4_模型A正式训练.md', '验收_T4_模型B重采样对照.md',
                 '诊断_T5_验证集失败分析.md', '运行_评测与验收.md', 'README.md']:
        assert (ROOT / name).is_file()
    record('Acceptance report and reusable run instructions present', sha(ROOT / '验收_T4-T7_完整实验报告.md'))
    result = dict(status='accepted', verified_at=datetime.now(timezone.utc).isoformat(),
                  scope='Experimental completion and evidence integrity; not a claim that the proposed method improves quality.',
                  requirements=requirements, archive_verification='Required next; recorded separately after archive creation.')
    destination = evaluation / 'final-acceptance.json'
    assert not destination.exists(), 'Preserve an existing acceptance decision.'
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(status='accepted', requirements=len(requirements), report=str(destination))))


if __name__ == '__main__':
    main()

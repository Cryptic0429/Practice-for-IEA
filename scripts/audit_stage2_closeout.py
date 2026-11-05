"""CPU-only final identity, saved-image, statistics and checkpoint audit.

Reads completed outputs; never trains, generates, changes selection or rewrites
the frozen statistics. PNG metrics are recomputed from the actual saved pixels.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path('/mnt/sdc/zhc/IEA')
OUT = ROOT / 'outputs/stage2-A-edit-grpo-v1'
sys.path.insert(0, str(ROOT / 'repo'))
from src.utils import compute_diff_score, load_and_resize_image
from evaluation_core import parse_parameters
from summarize_stage2 import bootstrap


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b)
    assert abs(a - b) < 1e-10, (a, b)


def main():
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(),
                  scope='CPU-only verification; no new inference, editor runs or training updates')
    gate = read(ROOT / 'configs/stage2-formal-gate.json')
    for p, digest in gate['evidence'].items():
        assert sha(ROOT / p) == digest, p
    report['frozen_gate_files'] = len(gate['evidence'])
    lock = read(ROOT / 'configs/stage2-selected-models.json')
    assert lock['status'] == 'frozen'
    for model, item in lock['models'].items():
        for filename, key in [('adapter_model.safetensors', 'adapter_sha256'),
                              ('adapter_config.json', 'adapter_config_sha256')]:
            assert sha(Path(item['adapter']) / filename) == item[key]
    for p, digest in lock['evaluation_identity'].items():
        assert sha(p) == digest
    report['locked_models_and_evaluation_identity'] = True
    checkpoints = []
    for folder in ('formal-v1', 'sft-control'):
        for step in (64, 128):
            audit = read(OUT / folder / f'checkpoint-{step}-audit.json')
            assert audit['passed'] and audit['step'] == step
            for p, digest in audit['files'].items():
                assert sha(OUT / folder / f'checkpoint-{step}' / p) == digest
            checkpoints.append(dict(folder=folder, step=step, files=len(audit['files'])))
        reload = read(OUT / folder / 'reload-verification.json')
        assert reload['status'] == 'passed' and reload['adapter_tensors_exact']
        assert reload['forward_exact'] and reload['max_logit_difference'] == 0
        assert reload['training_updates'] == 0
    report['checkpoint_state_hashes'] = checkpoints
    report['independent_reloads'] = True
    protocol = read(ROOT / 'configs/evaluation-protocol-v1.json')
    summary = read(OUT / 'summary.json')
    jobs = [(f'validation-GRPO-step{s}', 'validation', 'GRPO', s) for s in (64, 128)]
    jobs += [(f'{split}-{model}', split, model, None)
             for split in ('test_new', 'test_regression') for model in ('A', 'SFT', 'GRPO')]
    all_rows, reports = {}, {}
    for job, split, model, step in jobs:
        folder = OUT / 'evaluation' / job
        source = ROOT / 'data/stage2/stage2-A-edit-grpo-v1' / f'{split}.json'
        target = read(source)
        rows = [json.loads(x) for x in (folder / 'predictions.jsonl').read_text().splitlines()]
        status, run, aggregate = (read(folder / n) for n in ('status.json', 'run.json', 'summary.json'))
        assert status['status'] == 'completed' and status['completed'] == len(target) == len(rows)
        assert [r['id'] for r in rows] == [r['id'] for r in target]
        assert len({r['group_id'] for r in target}) == len(target)
        digest = (sha(OUT / 'formal-v1' / f'checkpoint-{step}' / 'adapter_model.safetensors')
                  if step else lock['models'][model]['adapter_sha256'])
        assert run['identity']['adapter_sha256'] == aggregate['adapter_sha256'] == digest
        assert aggregate['source_sha256'] == run['identity']['source_sha256'] == sha(source)
        if step is None:
            assert datetime.fromisoformat(lock['at']) < datetime.fromisoformat(run['started_at'])
            assert summary['prediction_sha256'][str(folder / 'predictions.jsonl')] == sha(folder / 'predictions.jsonl')
        unchanged, fallback, image_evidence = 0, 0, []
        for row in rows:
            parsed, score = row['parsing'], row['scoring']
            assert parse_parameters(row['response_text'], protocol) == parsed
            assert not score['metric_error'] and score['input_readable'] and score['reference_readable']
            original = load_and_resize_image(row['input_image'])
            reference = load_and_resize_image(row['reference_image'])
            edited = cv2.imread(score['edited_image_path'], cv2.IMREAD_COLOR)
            assert edited is not None and edited.shape == original.shape
            for arr, key in ((original, 'baseline_metrics'), (edited, 'edited_metrics')):
                computed = compute_diff_score(arr, reference)
                for metric, value in computed.items():
                    close(value, score[key][metric])
            same = np.array_equal(original, edited)
            unchanged += int(same)
            if score['fallback_used']:
                assert same and not score['editor_success']
                fallback += 1
            assert parsed['valid'] or score['fallback_used']
            image_evidence.append(dict(id=row['id'], path=score['edited_image_path'], sha256=sha(score['edited_image_path'])))
        for metric, key in [('pixel_L1', 'mean_l1'), ('pixel_L2', 'mean_rmse')]:
            close(float(np.mean([r['scoring']['edited_metrics'][metric] for r in rows])), aggregate[key])
        if step is None:
            m = summary['results'][split]['metrics'][model]
            assert m['rows'] == len(rows) and m['exact_identity_images'] == unchanged
            assert m['executable'] == sum(r['scoring']['editor_success'] for r in rows)
        reports[job] = dict(rows=len(rows), saved_png_metrics_exact=True,
                           parser_recheck_exact=True, fallback_identity_images=fallback,
                           identity_images=unchanged, predictions_sha256=sha(folder / 'predictions.jsonl'),
                           baseline_rmse=float(np.mean([r['scoring']['baseline_metrics']['pixel_L2'] for r in rows])),
                           saved_images=image_evidence)
        all_rows[job] = rows
    comparisons = 0
    for split in ('test_new', 'test_regression'):
        rows = {m: all_rows[f'{split}-{m}'] for m in ('A', 'SFT', 'GRPO')}
        for name, c in summary['results'][split]['comparisons'].items():
            candidate, reference = name.split('_vs_')
            for metric, expected in [('pixel_L1', c), ('pixel_L2', c['rmse_gain'])]:
                a = np.array([r['scoring']['edited_metrics'][metric] for r in rows[candidate]])
                b = np.array([r['scoring']['baseline_metrics' if reference == 'NoEdit' else 'edited_metrics'][metric]
                              for r in rows['A' if reference == 'NoEdit' else reference]])
                gain = b - a
                got = bootstrap(gain)
                for k, v in got.items():
                    assert np.allclose(v, expected[k], rtol=0, atol=1e-12), (split, name, metric, k)
                if metric == 'pixel_L1':
                    close(float(np.sort(gain)[-5:].sum()), c['five_largest_gains_sum'])
                    close(float(np.sort(gain)[:-5].mean()), c['mean_without_five_largest_gains'])
                    for kind, stats in c['by_instruction_type'].items():
                        mask = np.array([r['instruction_type'] == kind for r in rows[candidate]])
                        assert stats['n'] == int(mask.sum())
                        for k, v in bootstrap(gain[mask]).items():
                            assert np.allclose(v, stats[k], rtol=0, atol=1e-12)
                comparisons += 1
    selection = read(OUT / 'formal-v1/validation-selection.json')
    best = min(selection['validations'], key=lambda x: (x['mean_l1'], x['step']))
    assert best == selection['best'] and best['step'] == 128
    assert sha(OUT / 'formal-v1/selected/adapter_model.safetensors') == sha(Path(best['checkpoint']) / 'adapter_model.safetensors')
    report.update(passed=True, jobs=reports, validation_rows=400, test_rows=900,
                  paired_metric_comparisons_recomputed=comparisons,
                  selection_verified=True, checkpoint_limitation='State integrity audit, not an executed resume equivalence test.')
    path = OUT / 'closeout-numerical-audit.json'
    with path.open('x') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k != 'jobs'}))


if __name__ == '__main__':
    main()

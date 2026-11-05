"""Requirement-level current-state audit. A preaudit cannot declare completion."""
import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path('/mnt/sdc/zhc/IEA')
OUT = ROOT / 'outputs/goal-v2-20261103'
ART = ROOT / 'artifacts/goal-v2-20261103'
PACK = ROOT / 'artifacts/goal-v2-20261103-packaging'

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verification', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    assert not a.output.exists()
    checks = []
    def record(requirement, passed, evidence, limits=None):
        checks.append({'requirement': requirement, 'proved': bool(passed), 'evidence': evidence,
                       'limits': limits or []})
    lock = read(OUT / 'final-models-lock.json')
    protocol = read(OUT / 'final-protocol.json')
    opened = read(OUT / 'test-opened.json')
    analysis = read(OUT / 'final-analysis/summary.json')
    manifest = read(PACK / 'project/MANIFEST.json')
    member = {row['path']: row for row in manifest['files']}
    def bundled(path):
        relative = str(Path(path).relative_to(ROOT))
        return relative in member and sha(path) == member[relative]['sha256']
    code_ok = all(sha(ROOT / p) == digest and bundled(ROOT / p) for p, digest in lock['code_hashes'].items())
    record('Frozen inference/editor/metric code and model identities', code_ok and
        sha(OUT / 'final-models-lock.json') == analysis['model_lock_sha256'] and
        sha(OUT / 'final-protocol.json') == lock['final_protocol_sha256'],
        {'model_lock_sha256': sha(OUT / 'final-models-lock.json'), 'checked_code_files': len(lock['code_hashes'])})
    times = [datetime.datetime.fromisoformat(lock['locked_at']), datetime.datetime.fromisoformat(opened['opened_at'])]
    record('Model and protocol locked before independent test opened', times[0] < times[1] and
        opened['model_lock_sha256'] == sha(OUT / 'final-models-lock.json') and
        opened['protocol_sha256'] == sha(OUT / 'final-protocol.json'),
        {'locked_at': lock['locked_at'], 'opened_at': opened['opened_at'],
         'test_policy': lock['test_policy']})
    data = ROOT / 'data/goal-v2-20261103'
    train = read(data / 'train3k-contrast-calibrated.json')
    validation = read(data / 'validation.json')
    test = read(data / 'test_sealed.json')
    groups = [set(row['group_id'] for row in rows) for rows in [train, validation, test]]
    disjoint = all(not groups[i] & groups[j] for i in range(3) for j in range(i+1, 3))
    asset_paths = {Path(row['images'][0]) for rows in [train, validation, test] for row in rows}
    asset_paths |= {Path(row['extra_info']['reference_image_path']) for rows in [train, validation, test] for row in rows}
    record('Independent71 source groups, no training/validation overlap, complete data assets',
        len(train) == 3000 and len(groups[0]) == 1078 and len(test) == len(groups[2]) == 71 and
        len(validation) == len(groups[1]) == 200 and disjoint and
        sha(data / 'test_sealed.json') == lock['test_source_sha256'] and all(bundled(p) for p in asset_paths),
        {'group_counts': [len(g) for g in groups], 'disjoint': disjoint, 'bundled_unique_images': len(asset_paths),
         'test_source_sha256': sha(data / 'test_sealed.json')},
        ['Project-record and conservative perceptual screening, not upstream-pretraining or full semantic-duplicate certification.'])
    ids = [r['id'] for r in test]
    predictions = {}
    digests = {}
    for name, entry in lock['models'].items():
        directory = OUT / 'final-evaluation' / name
        p = directory / 'predictions.jsonl'
        rows = [json.loads(line) for line in p.open()]
        assert [r['id'] for r in rows] == ids and [r['group_id'] for r in rows] == [r['group_id'] for r in test]
        assert read(directory / 'status.json')['status'] == 'completed'
        assert read(directory / 'run.json')['identity']['model_lock_sha256'] == sha(OUT / 'final-models-lock.json')
        assert sha(Path(entry['path']) / 'adapter_model.safetensors') == entry['adapter_sha256']
        assert sha(Path(entry['path']) / 'adapter_config.json') == entry['config_sha256']
        assert bundled(p) and all(bundled(Path(r['scoring']['edited_image_path'])) for r in rows)
        predictions[name] = rows
        digests[name] = sha(p)
    baseline = min(lock['baseline_keys'], key=lambda n: np.mean([r['scoring']['edited_metrics']['pixel_L1'] for r in predictions[n]]))
    base_l1 = np.array([r['scoring']['edited_metrics']['pixel_L1'] for r in predictions[baseline]])
    rng = np.random.default_rng(20261103)
    indices = rng.integers(0, 71, (10000, 71))
    gains = {}
    for name in lock['candidate_keys']:
        candidate = np.array([r['scoring']['edited_metrics']['pixel_L1'] for r in predictions[name]])
        delta = base_l1-candidate
        ci = np.quantile(delta[indices].mean(1), [.025, .975])
        relative = float(delta.mean()/base_l1.mean())
        assert relative >= .03 and ci[0] > 0
        assert abs(relative-analysis['per_seed_comparisons'][name][baseline]['relative_mean_gain']) < 1e-12
        gains[name] = {'l1': float(candidate.mean()), 'relative_gain': relative, 'paired_gain_ci95': ci.tolist()}
        for other in lock['baseline_keys']:
            other_l1 = np.array([r['scoring']['edited_metrics']['pixel_L1'] for r in predictions[other]])
            assert float((other_l1-candidate).mean()/other_l1.mean()) >= .03
    record('At least3% paired mean L1 improvement for all3 seeds against every original baseline',
        sorted(lock['training_seeds']) == [20261103, 20261104, 20261105] and baseline == analysis['strongest_original_baseline'],
        {'strongest_original': baseline, 'per_seed': gains, 'bootstrap': '10000 shared71-source-group resamples; no213-row pooling'},
        ['Positive per-seed intervals; not every interval lower bound exceeds3%.', 'Unadjusted per-comparison intervals; no universal aesthetic claim.'])
    image_checks = read(OUT / 'final-analysis/image-diagnostics.json')
    record('All497 saved image metrics replayed and coverage/no-op guards retained',
        image_checks['predictions_sha256'] == digests and len(image_checks['rows']) == 71 and
        all(analysis['models'][n]['max_saved_pixel_metric_replay_error'] == 0 for n in predictions) and
        all(all(r['scoring']['editor_success'] and not r['empty_parameters'] and not r['all_zero_parameters'] for r in predictions[n]) for n in lock['candidate_keys']) and
        all(analysis['models'][n]['pixel_identical'] == 0 and
            analysis['subset_robustness'][n]['positive_noop_gain_removed']['relative_gain'] >= .03 and
            analysis['subset_robustness'][n]['positive_noop_gain_removed']['ci95_gain'][0] > 0 for n in lock['candidate_keys']),
        {'models': 7, 'predictions': 497, 'each_candidate_executable': 71,
         'near_identity_per_seed': {n: analysis['models'][n]['near_identity'] for n in lock['candidate_keys']},
         'metric_function': 'Original frozen compute_diff_score; matching prediction-SHA diagnostics and every savedPNG SHA in archive manifest.'})
    training = {}
    for name in lock['candidate_keys']:
        path = Path(lock['models'][name]['path'])
        average = read(path / 'averaging.json')
        sources = [Path(s['path']) for s in average['sources']]
        assert sources[0].parent == sources[1].parent
        assert [s.name for s in sources] == ['checkpoint-512', 'checkpoint-1024']
        x, y = [load_file(str(p / 'adapter_model.safetensors')) for p in sources]
        z = load_file(str(path / 'adapter_model.safetensors'))
        assert len(z) == 112 and x.keys() == y.keys() == z.keys()
        assert all(torch.equal(z[k], (x[k].float()+y[k].float())*.5) for k in z)
        parent = sources[0].parent
        settings = read(parent / 'status.json')['settings']
        assert settings['seed'] == lock['models'][name]['seed']
        config = read(parent / 'run-config.json')
        assert config['max_steps'] == 1024 and config['learning_rate'] == .0001 and config['gradient_accumulation_steps'] == 4
        for checkpoint in sources:
            assert read(checkpoint / 'trainer_state.json')['global_step'] == int(checkpoint.name.split('-')[-1])
            for f in ['adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json']:
                assert bundled(checkpoint / f)
        checkpoint_audit = read(parent / 'checkpoint-1024-audit.json')
        assert checkpoint_audit['passed'] and checkpoint_audit['optimizer_parameter_states'] == 112
        for f, digest in checkpoint_audit['files'].items():
            assert sha(sources[1] / f) == digest
        training[name] = {'seed': settings['seed'], 'updates': 1024, 'checkpoints': [str(p) for p in sources],
                          'fixed_mean112_tensors_exact': True, 'optimizer_states': 112}
    record('Three independent training seeds, fixed recipe/export and restorable raw checkpoints', True, training,
           ['Averaged inference weights are not paired with raw optimizer state; use raw checkpoint for continuation.'])
    selected = min(lock['candidate_keys'], key=lambda n: lock['models'][n]['validation_l1'])
    replay = read(OUT / 'delivery-inference-replay-v1/result.json')
    smoke = read(OUT / 'checkpoint-restore-smoke-v1/summary.json')
    record('Best model selected by validation, exact delivered inference replay and executed restore smoke',
        selected == lock['best_by_validation'] == 'Seed20261104' and all(replay['replay_checks'].values()) and
        smoke['passed'] and smoke['restored_optimizer_step'] == 512 and smoke['executed_optimizer_step'] == 513 and
        smoke['chosen_inference_adapter_unchanged_sha256'] == lock['models'][selected]['adapter_sha256'],
        {'selected': selected, 'inference_replay': replay['replay_checks'], 'restore_steps': [512, 513]},
        ['Restore uses fixed4 training rows, not exact historical next-dataloader replay.'])
    visual = read(OUT / 'final-visual-review-summary.json')
    receipt = read(ART / 'blind-review/observations-freeze-receipt.json')
    explicit = {n: visual['models'][n]['instruction'] for n in lock['models']}
    record('Instruction safeguards and all30 preselected anonymous cases reviewed before decoding',
        visual['outputs'] == 210 and visual['cases'] == 30 and
        sha(ART / 'blind-review/observations-before-decoding.json') == receipt['observations_sha256'] == visual['observations_sha256'] and
        all(analysis['models'][n]['all_assessed_directions_aligned'] >= analysis['models'][baseline]['all_assessed_directions_aligned'] for n in lock['candidate_keys']) and
        all(explicit[n].get('aligned', 0) >= explicit[baseline].get('aligned', 0) and
            explicit[n].get('opposite', 0) <= explicit[baseline].get('opposite', 0) for n in lock['candidate_keys']),
        {'explicit10_counts': explicit, 'generic_unassessable': 20, 'observations_sha256': receipt['observations_sha256'],
         'global_proxy_assessable': 9}, visual['limits'] +
        ['Some instruction subtypes regress; B has3/10 fully aligned versus2/10 each new seed. No complete local-edit capability or statistical semantic noninferiority established.'])
    noedit = np.mean([r['scoring']['baseline_metrics']['pixel_L1'] for r in predictions[baseline]])
    record('NoEdit honestly reported without claiming stable superiority over it',
        abs(noedit-analysis['NoEdit_l1']) < 1e-12 and all(noedit < v['l1'] for v in gains.values()),
        {'NoEdit_l1': float(noedit), 'selected_l1': gains[selected]['l1'],
         'conclusion': 'Stable improvement over original self-trained baselines; no demonstrated advantage over NoEdit or general aesthetics.'})
    base = read(OUT / 'base-source-current-audit.json')
    base_ok = all(sha(Path(r['path'])) == r['sha256'] for r in base['files'])
    record('Frozen environment and external base identity verified and disclosed',
        base['passed'] and base_ok and bundled(OUT / 'runtime-packages-current.json') and
        bundled(OUT / 'requirements-frozen.txt') and bundled(OUT / 'environment-system-current.json'),
        {'base_files_rehashed_now': len(base['files']), 'base_revision': manifest['external_dependency']['revision'],
         'base_weights_in_archive': False, 'packages': len(read(OUT / 'runtime-packages-current.json')['packages'])},
        ['Base shards and venv binaries external; not a standalone installed offline environment.'])
    negative = [OUT / 'evaluation/validation-json-calibrated-locked256-lr1e-4-s20261104/summary.json',
                OUT / 'evaluation/validation-json-calibrated-locked256-lr1e-4-s20261105/summary.json',
                OUT / 'evaluation/validation-json3k-qualityweighted512-lr1e-4-s20261103/summary.json',
                OUT / 'evaluation/validation-json-memorize64-long512-lr1e-4-s20261103/summary.json']
    record('Complete success and negative findings preserved', all(bundled(p) for p in negative),
           {str(p.relative_to(ROOT)): read(p)['mean_l1'] for p in negative})
    resources = read(OUT / 'resource-account.json')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).split()
    record('At most3 campaign GPUs and protected GPU1 untouched; GPU work terminal',
        set(resources['gpu_hours_covered_by_telemetry']) == {'0', '2', '3'} and
        resources['maximum_distinct_campaign_gpus'] == 3 and active == ['3378799'],
        {'telemetry_covered_gpu_hours': resources['summed_gpu_hours_covered'], 'active_gpu_pids': active,
         'protected_existing_pid': 3378799}, resources['limits'])
    archive_receipt = read(PACK / 'archive-receipt.json')
    verified = read(a.verification) if a.verification and a.verification.exists() else None
    archive_ok = verified is not None and verified['passed'] and verified['archive_sha256'] == archive_receipt['archive_sha256'] and verified['verified_members'] == manifest['file_count']
    record('Full archive downloaded and every member independently verified locally', archive_ok,
           verified or {'status': 'pending_local_download_and_verification', 'archive_bytes': archive_receipt['archive_bytes']})
    report = {'status': 'complete_requirements_proved' if all(c['proved'] for c in checks) else 'incomplete',
              'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'requirements': checks,
              'all_requirements_proved': all(c['proved'] for c in checks),
              'pending_requirements': [c['requirement'] for c in checks if not c['proved']],
              'archive_sha256': archive_receipt['archive_sha256'], 'selected': selected,
              'objective_scope': 'At least3% paired mean L1 improvement versus original best self-trained baseline,3 training seeds, execution/instruction safeguards, NoEdit report and full delivery; no expanded claim of universal editing quality.'}
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'requirements'}, ensure_ascii=False))

if __name__ == '__main__':
    main()

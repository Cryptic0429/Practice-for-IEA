"""Commit the formal configuration only after measured resource/reload checks pass."""
import copy
import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ROOT=Path('/mnt/sdc/zhc/IEA')


def main():
    profiles={name:json.loads((ROOT/f'outputs/benchmarks/{name}/benchmark.json').read_text())
              for name in ['pilot-65k-r8','pilot-262k-b2-r8','stress-262k-b2-r8']}
    chosen=profiles['pilot-262k-b2-r8']
    stress=profiles['stress-262k-b2-r8']
    reload=json.loads((ROOT/'outputs/benchmarks/pilot-262k-b2-r8/reload-verification.json').read_text())
    assert reload['adapter_reload_verified']
    assert chosen['measured_samples']>=160 and chosen['measured_optimizer_steps']>=10
    assert chosen['all_training_losses_finite'] and stress['all_training_losses_finite']
    assert stress['actual_training_samples']>=32
    for field in ['image_max_pixels','lora_rank','lora_target','per_device_train_batch_size','gradient_accumulation_steps']:
        assert chosen['config'][field]==stress['config'][field]
    peak=max(chosen['training_peak_device_mib'],stress['training_peak_device_mib'])
    assert chosen['device_total_mib']-peak>=4096, 'Insufficient measured VRAM margin'
    manifest_path=ROOT/'data/stage1/splits/image-edit-3k-v1/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    sequences=json.loads((ROOT/'logs/frozen-sequence-audit.json').read_text())
    assert all(s['truncated_rows']==0 for s in sequences['summary'].values())
    config=copy.deepcopy(chosen['config'])
    config.update(dataset='iea_edit_3k_train',eval_dataset='iea_edit_3k_validation',
                  output_dir=str(ROOT/'outputs/stage1-A-image-edit-3k-v1'),
                  do_eval=True,per_device_eval_batch_size=2,prediction_loss_only=True,
                  max_steps=376,learning_rate=1e-4,lr_scheduler_type='cosine',warmup_ratio=0.05,
                  max_grad_norm=1.0,logging_steps=5,logging_first_step=True,
                  eval_strategy='steps',eval_steps=94,save_strategy='steps',save_steps=94,
                  save_total_limit=2,save_only_model=False,load_best_model_at_end=True,
                  metric_for_best_model='eval_loss',greater_is_better=False,
                  dataloader_drop_last=False,disable_tqdm=False)
    assert not config.get('adapter_name_or_path')
    file=ROOT/'configs/qlora-stage1-A-3k.yaml'
    header=('# Frozen Image-Edit pilot A; original Qwen + 3000 train / 200 validation.\n'
            '# Test data is never passed to Trainer. Pilot adapters are not reused.\n'
            '# Two complete passes: 188 updates/pass, including one 8-sample tail batch.\n'
            '# Explicit 376 steps avoids Transformers 4.51.3 epoch-step floor rounding.\n')
    file.write_text(header+yaml.safe_dump(config,sort_keys=False))
    pure=6000/chosen['samples_per_second']
    decision=dict(timestamp=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),
                  formal_config=str(file),formal_config_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                  split_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  model_revision=chosen['model_revision'],
                  repository_commit=subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip(),
                  selected_profile='pilot-262k-b2-r8',
                  reason='Higher image resolution fits with over 4 GiB device memory margin, including longest training records; throughput fits the pilot budget. This is an engineering choice, not evidence of better image-editing quality.',
                  training_rows=3000,validation_rows=200,test_rows=200,
                  planned_data_passes=2,planned_sample_exposures=6000,optimizer_steps=376,
                  effective_batch_size=16,tail_batch_size_per_pass=8,
                  evaluation_and_checkpoint_steps=[94,188,282,376],
                  minimum_free_gpu_mib=math.ceil(peak+2048),
                  measured_profiles={name:{k:r[k] for k in ['samples_per_second','training_peak_allocated_gib',
                      'training_peak_reserved_gib','training_peak_device_mib','actual_optimizer_steps',
                      'actual_training_samples','measured_samples']} for name,r in profiles.items()},
                  maximum_observed_device_mib=peak,remaining_device_memory_mib=chosen['device_total_mib']-peak,
                  estimated_pure_training_minutes=pure/60,
                  estimated_wall_budget_minutes=[(pure*1.15+chosen['setup_seconds'])/60,(pure*1.4+chosen['setup_seconds'])/60],
                  estimate_note='Extrapolation from 160 post-warmup samples. Wall budget adds heuristic allowance for setup, validation, saves and variability; it is not a completed full-run timing.',
                  adapter_reload=reload,formal_training_started=False,
                  limitations=['Benchmark compares complete configurations, changing image resolution and micro-batch together.',
                               'Training length and learning rate are a reproducible starting recipe, not quality-tuned optima.',
                               'No final test model evaluation has been performed.'])
    (ROOT/'configs/stage1-A-3k-decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(json.dumps(decision,indent=2),flush=True)


if __name__=='__main__':main()

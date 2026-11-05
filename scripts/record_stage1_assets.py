"""Record validated asset paths; does not start training."""
import json
import hashlib
from pathlib import Path

ROOT=Path('/mnt/sdc/zhc/IEA')
data_report=json.loads((ROOT/'logs/stage1-data-readiness.json').read_text())
model_report=json.loads((ROOT/'logs/qwen-stage1-readiness.json').read_text())
revision=json.loads((ROOT/'configs/qwen-base-revision.json').read_text())
registry=json.loads((ROOT/'data/dataset_info.json').read_text())
datasets={name:dict(file_name=value['file_name'],sha256=hashlib.sha256(Path(value['file_name']).read_bytes()).hexdigest())
          for name,value in registry.items() if name.startswith('gier_sft_')}
assert model_report['original_qwen_loaded'] and model_report['training_steps']==0
assert data_report['missing_or_corrupt_images']==0 and data_report['invalid_labels']==0
assets=dict(status='ready_for_grouped_split_and_training_budget_probe',
            model_name_or_path=revision['local_path'],model_revision=revision['revision'],
            dataset_dir=str(ROOT/'data'),image_edit_dataset='gier_sft_distill',
            image_summary_dataset='gier_sft_summary',datasets=datasets,
            reports=[str(ROOT/'logs'/name) for name in ['stage1-data-readiness.json','qwen-stage1-readiness.json','environment-readiness.json']],
            next_steps=['Freeze train/validation/test groups and training subset.',
                        'Measure real multi-step throughput and peak training VRAM.',
                        'Choose final training settings; no formal training has started.'])
(ROOT/'configs/stage1-assets.json').write_text(json.dumps(assets,indent=2)+'\n')
print(json.dumps(assets,indent=2))

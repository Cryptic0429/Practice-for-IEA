"""Fail before launching if the frozen data or approved training inputs changed."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml

ROOT=Path('/mnt/sdc/zhc/IEA')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--require-free-gpu',action='store_true')
    cli=parser.parse_args()
    decision=json.loads((ROOT/'configs/stage1-A-3k-decision.json').read_text())
    config_path=Path(decision['formal_config'])
    assert hashlib.sha256(config_path.read_bytes()).hexdigest()==decision['formal_config_sha256'], 'Formal config changed'
    config=yaml.safe_load(config_path.read_text())
    manifest_path=ROOT/'data/stage1/splits/image-edit-3k-v1/manifest.json'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==decision['split_manifest_sha256']
    manifest=json.loads(manifest_path.read_text())
    for item in manifest['files'].values():
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
    groups_path=manifest_path.parent/'group-assignments.json'
    assert hashlib.sha256(groups_path.read_bytes()).hexdigest()==manifest['group_assignment_sha256']
    registry=json.loads((ROOT/'data/dataset_info.json').read_text())
    assert config['dataset']=='iea_edit_3k_train'
    assert config['eval_dataset']=='iea_edit_3k_validation'
    assert registry[config['dataset']]['file_name']==manifest['files']['train']['path']
    assert registry[config['eval_dataset']]['file_name']==manifest['files']['validation']['path']
    assert config['model_name_or_path']==str(ROOT/'models/Qwen2.5-VL-7B-Instruct')
    assert not config.get('adapter_name_or_path'), 'Formal run must start from the original Qwen'
    assert config['freeze_vision_tower'] and config['freeze_multi_modal_projector']
    revision=json.loads((ROOT/'configs/qwen-base-revision.json').read_text())['revision']
    assert revision==decision['model_revision']
    commit=subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()
    assert commit==decision['repository_commit']
    if cli.require_free_gpu:
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES','2')
        assert ',' not in gpu, 'This configuration was validated on one GPU'
        free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        required=decision['minimum_free_gpu_mib']
        if free<required: raise RuntimeError(f'GPU {gpu} has {free} MiB free; at least {required} MiB is required. Choose an available GPU with IEA_GPU.')
    print('FROZEN_TRAINING_PREFLIGHT_PASSED: train=3000 validation=200 test=200; model, code and data identities match')


if __name__=='__main__':main()

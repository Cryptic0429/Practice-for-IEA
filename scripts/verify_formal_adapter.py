"""Reload the selected formal adapter; one training probe, no updates or test use.

Run after the supervisor reports completed:
IEA_GPU=2 bash -c 'source /mnt/sdc/zhc/IEA/activate.sh; python /mnt/sdc/zhc/IEA/scripts/verify_formal_adapter.py'
"""
import argparse
import fcntl
import json
import math
import subprocess
from pathlib import Path
import supervise_stage1_training as monitor

import torch
import yaml
from peft import get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import set_seed
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer, SFTDataCollatorWith4DAttentionMask
from benchmark_stage1 import eval_probe
from supervise_stage1_training import ROOT, RUN, OUT, stamp, write_json, verify_artifacts


def main():
    global RUN, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=['A', 'B'], default='A')
    parser.add_argument('--check-only', action='store_true')
    cli = parser.parse_args()
    if cli.experiment == 'B':
        RUN = monitor.RUN = ROOT / 'logs/stage1-B-3k'
        OUT = monitor.OUT = ROOT / 'outputs/stage1-B-image-edit-3k-v1'
        subprocess.run([str(ROOT / 'envs/iea/bin/python'), str(ROOT / 'scripts/verify_stage1_B.py')], check=True)
    subprocess.run([str(ROOT / 'scripts/train_stage1_frozen.sh'), '--check-only'], check=True)
    config = yaml.safe_load((ROOT / f'configs/qlora-stage1-{cli.experiment}-3k.yaml').read_text())
    probe_dataset = 'iea_edit_3k_B_train' if cli.experiment == 'B' else 'iea_edit_3k_train'
    assert config['dataset'] == probe_dataset
    config.update(adapter_name_or_path=str(OUT), max_samples=1,
                  output_dir=str(RUN / 'reload-parser-output'), do_eval=False,
                  eval_strategy='no', save_strategy='no', load_best_model_at_end=False)
    config.pop('eval_dataset')
    ma, da, ta, fa, ga = get_train_args(config)
    assert da.eval_dataset is None
    if cli.check_only:
        print('FORMAL_RELOAD_ARGUMENTS_PASSED: one training probe, no optimizer or test access')
        return
    lock = (RUN / 'reload.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report_path = RUN / 'reload-verification.json'
    artifact = verify_artifacts()
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        assert previous['adapter_sha256'] == artifact['adapter_sha256']
        assert previous['adapter_reload_verified']
        artifact['full_model_reload_verified'] = True
        artifact['reload_verified_at'] = previous['verified_at']
        write_json(RUN / 'artifact-verification.json', artifact)
        print('ALREADY_VERIFIED', json.dumps(previous))
        return
    status = json.loads((RUN / 'status.json').read_text())
    assert status['status'] == 'completed', status['status']
    subprocess.run([str(ROOT / 'envs/iea/bin/python'),
                    str(ROOT / 'scripts/verify_frozen_training_config.py'),
                    '--require-free-gpu'], check=True)
    set_seed(ta.seed)
    tm = load_tokenizer(ma)
    template = get_template_and_fix_tokenizer(tm['tokenizer'], da)
    ds = get_dataset(template, ma, da, ta, stage='sft', **tm)['train_dataset']
    assert len(ds) == 1
    # Match the training precision/upcasts, without constructing an optimizer.
    model = load_model(tm['tokenizer'], ma, fa, is_trainable=True)
    state = get_peft_model_state_dict(model)
    saved = load_file(str(OUT / 'adapter_model.safetensors'))
    assert set(state) == set(saved)
    for name, tensor in state.items():
        assert torch.equal(tensor.detach().cpu(), saved[name]), name
    collator = SFTDataCollatorWith4DAttentionMask(
        template=template, model=model, pad_to_multiple_of=8, label_pad_token_id=-100,
        attn_implementation='sdpa', compute_dtype=ma.compute_dtype, **tm)
    probe = eval_probe(model, collator, ds)
    assert math.isfinite(probe['loss']) and all(math.isfinite(v) for v in probe['logits'])
    report = dict(verified_at=stamp(), adapter_reload_verified=True,
                  adapter_tensors_exactly_equal=len(saved),
                  adapter_sha256=artifact['adapter_sha256'],
                  best_checkpoint=artifact['best_checkpoint'],
                  training_updates_during_reload=0, probe_dataset=probe_dataset,
                  probe_samples=1, probe=probe, test_set_evaluated=False,
                  scope='Saved tensors match reloaded tensors and a training-probe forward is finite; this is not an image quality evaluation.')
    write_json(report_path, report)
    artifact['full_model_reload_verified'] = True
    artifact['reload_verified_at'] = report['verified_at']
    write_json(RUN / 'artifact-verification.json', artifact)
    print('FORMAL_ADAPTER_RELOAD_PASSED', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

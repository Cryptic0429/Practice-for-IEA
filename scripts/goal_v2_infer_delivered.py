"""Replay a delivered model on one frozen row using the exact final inference path.

Reference/labels are removed before generation and used only in optional scoring.
The output must be new, so historical evaluation artifacts cannot be overwritten.
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--id', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--gpu', choices=['0', '2', '3'], required=True)
    ap.add_argument('--model', default='Seed20261104')
    ap.add_argument('--expected-prediction', type=Path)
    a = ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == a.gpu
    assert not a.output.exists()
    memory = int(subprocess.check_output(['nvidia-smi', '-i', a.gpu,
                 '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True))
    assert memory < 512, 'Do not use an occupied GPU'
    import torch
    from transformers import set_seed
    from trl.models.utils import peft_module_casting_to_bf16
    from stage2_runtime import ROOT, processor, encoded, load_model
    from goal_v2_final_evaluate import clean
    from evaluation_core import sample_seed, parse_parameters, Scorer
    out = ROOT / 'outputs/goal-v2-20261103'
    lock = json.loads((out / 'final-models-lock.json').read_text())
    entry = lock['models'][a.model]
    adapter = Path(entry['path'])
    assert sha(adapter / 'adapter_model.safetensors') == entry['adapter_sha256']
    assert sha(adapter / 'adapter_config.json') == entry['config_sha256']
    for path, digest in lock['code_hashes'].items():
        assert sha(ROOT / path) == digest, path
    rows = json.loads(a.dataset.read_text())
    row = next(r for r in rows if r['id'] == a.id)
    torch.set_num_threads(4)
    p = processor()
    model = load_model(adapter, trainable=True)
    peft_module_casting_to_bf16(model)
    for weight in model.parameters():
        weight.requires_grad_(False)
    model.eval()
    set_seed(sample_seed(row['id'], 20261101))
    inputs = {k: v.to('cuda') for k, v in encoded(p, clean(row, entry['format'])).items()}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        result = model.generate(**inputs, do_sample=False, max_new_tokens=512,
            repetition_penalty=1., pad_token_id=p.tokenizer.pad_token_id,
            eos_token_id=p.tokenizer.eos_token_id, use_cache=True, disable_compile=True)
    tokens = result[0, inputs['input_ids'].shape[1]:].tolist()
    response = p.decode(tokens, skip_special_tokens=True)
    protocol = json.loads((ROOT / 'configs/evaluation-protocol-v1.json').read_text())
    parsed = parse_parameters(response, protocol)
    a.output.mkdir(parents=True, exist_ok=False)
    scoring = Scorer(ROOT, protocol).score(row, parsed, a.output / 'edited.png')
    report = {'id': row['id'], 'model': a.model,
              'adapter_sha256': entry['adapter_sha256'], 'response_text': response,
              'response_length': len(tokens), 'parsing': parsed, 'scoring': scoring,
              'edited_image_sha256': sha(a.output / 'edited.png'),
              'input_contract': 'Only original image, system and user request; references/labels excluded before generation.'}
    if a.expected_prediction:
        expected = next(json.loads(line) for line in a.expected_prediction.open()
                        if json.loads(line)['id'] == a.id)
        checks = {
            'response_exact': response == expected['response_text'],
            'parameter_exact': parsed == expected['parsing'],
            'edited_png_exact': report['edited_image_sha256'] == expected['edited_image_sha256'],
            'l1_exact': scoring['edited_metrics']['pixel_L1'] == expected['scoring']['edited_metrics']['pixel_L1']}
        report['replay_checks'] = checks
        assert all(checks.values()), checks
    (a.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['id', 'model', 'edited_image_sha256']} |
                     {'replay_checks': report.get('replay_checks')}))

if __name__ == '__main__':
    main()

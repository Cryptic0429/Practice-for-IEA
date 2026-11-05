"""Compare frozen HF input tensors without loading model weights, CPU only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.chat.hf_engine import HuggingfaceEngine

ROOT = Path('/mnt/sdc/zhc/IEA')


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--full-test-inputs', action='store_true',
                     help='After all predictions are fixed, audit all 200 test inputs; no forward calls.')
    args_cli = cli.parse_args()
    suffix = 'input-tensor-parity-test-audit.json' if args_cli.full_test_inputs else 'input-tensor-parity-audit.json'
    output = ROOT / 'outputs/evaluation-v1' / suffix
    assert not output.exists(), 'Preserve the existing input audit.'
    protocol_path = ROOT / 'configs/evaluation-protocol-v1.json'
    p = json.loads(protocol_path.read_text())
    split = 'test' if args_cli.full_test_inputs else 'train'
    source = Path(p['split_directory']) / f'{split}.json'
    manifest = json.loads(source.with_name('manifest.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['files'][split]['sha256']
    rows = json.loads(source.read_text())
    if args_cli.full_test_inputs:
        assert len(rows) == 200
        for name in p['models']:
            status = json.loads((ROOT / f'outputs/evaluation-v1/test-{name}-full/status.json').read_text())
            assert status['status'] == 'completed' and status['completed'] == 200
    else:
        rows = rows[:8]
    torch.set_num_threads(2)
    fake_model = SimpleNamespace(device=torch.device('cpu'), dtype=torch.bfloat16,
                                 config=SimpleNamespace(model_type='qwen2_5_vl'))
    conditions, processors = {}, {}
    for name in ['Base', 'Official']:
        args = {**p['model_common'], **p['models'][name], **p['generation']}
        args.pop('num_return_sequences', None)
        ma, da, fa, ga = get_infer_args(args)
        module = load_tokenizer(ma)
        tokenizer, processor = module['tokenizer'], module['processor']
        tokenizer.padding_side = 'left'
        template = get_template_and_fix_tokenizer(tokenizer, da)
        processors[name] = dict(tokenizer_class=type(tokenizer).__name__,
                                 processor_class=type(processor).__name__,
                                 image_processor_class=type(processor.image_processor).__name__,
                                 stop_token_ids=template.get_stop_token_ids(tokenizer))
        encoded = []
        for row in rows:
            messages = [dict(role=m['role'], content=m['content']) for m in row['conversations'] if m['role'] == 'user']
            system = next(m['content'] for m in row['conversations'] if m['role'] == 'system')
            inputs, length = HuggingfaceEngine._process_args(
                fake_model, tokenizer, processor, template, ga.to_dict(), messages,
                system=system, images=row['images'], input_kwargs=dict(p['generation']))
            encoded.append((inputs, length))
        conditions[name] = encoded
    checks = []
    for row, (base, base_len), (official, official_len) in zip(rows, conditions['Base'], conditions['Official']):
        assert set(base) == set(official)
        tensors = {}
        for key, left in base.items():
            right = official[key]
            if torch.is_tensor(left):
                equal = torch.equal(left, right)
                difference = float((left.float()-right.float()).abs().max()) if left.shape == right.shape else None
                tensors[key] = dict(shape=list(left.shape), dtype=str(left.dtype), exactly_equal=equal,
                                    max_absolute_difference=difference)
        same_generation = base['generation_config'].to_dict() == official['generation_config'].to_dict()
        checks.append(dict(id=row['id'], prompt_length=base_len,
                           same_prompt_length=base_len == official_len, tensors=tensors,
                           same_generation_configuration=same_generation,
                           passed=base_len == official_len and same_generation and all(t['exactly_equal'] for t in tensors.values())))
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), source_split=split, sample_count=len(rows),
                  protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                  loaded_model_weights=False, model_forward_calls=0, processors=processors, probes=checks,
                  all_probes_identical=all(x['passed'] for x in checks),
                  scope='Reconstructed inputs through the frozen HF message/token/image preprocessing path and generation settings, on CPU without model weights. Base/A/B share the same base tokenizer/processor; Official compared here. BaseSchema intentionally adds its declared prompt suffix. This covers only the listed inputs, not every possible future input.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(path=str(output), all_probes_identical=result['all_probes_identical'], processors=processors,
                         mismatches=[x for x in checks if not x['passed']]), indent=2))
    if not result['all_probes_identical']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

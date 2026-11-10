"""Fresh, matched language LoRA on audited original or author full weights."""
from pathlib import Path
import torch
from functional_v1.common import ROOT, BASE, read, sha
from functional_v1.runtime import processor, encode, generate, verify_adapter
from functional_v7.runtime import build_batch, create_adapter, set_trainable, predict
from functional_v7.outcomes import input_row, target_text, parse

EXP = ROOT / 'experiments/functional-v9-20261106'
DATA = ROOT / 'data/training/functional-v7-20261105'
ORIGINS = {'base': BASE, 'author': ROOT / 'models/Qwen2.5-VL-7B-IEA'}

def verify_origin(origin):
    assert origin in ORIGINS
    audit = read(EXP / 'origin-audit.json')
    identity = audit['origins'][origin]
    assert ROOT / identity['path'] == ORIGINS[origin]
    for item in identity['shards']:
        stat = (ROOT / item['path']).stat()
        assert stat.st_size == item['bytes'] and stat.st_mtime_ns == item['mtime_ns'], 'Audited weight file changed'
    for name, digest in identity['assets'].items():
        assert sha(ORIGINS[origin] / name) == digest
    return identity


def load_model(origin, adapter=None, train=False, seed=None):
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, set_seed
    from peft import PeftModel, prepare_model_for_kbit_training, get_peft_model_state_dict
    verify_origin(origin)
    assert not train or (adapter is None and seed is not None), 'Fresh single adapter only'
    base, information = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ORIGINS[origin], torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 0},
        output_loading_info=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    assert not any(information.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), information
    if train:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False})
        set_seed(seed)  # Identical fresh A matrices despite different full-weight contents.
        model = create_adapter(base, 'language')
        initial = get_peft_model_state_dict(model)
        assert len(initial) == 112 and all('visual' not in key for key in initial)
        assert sum('lora_B' in key for key in initial) == 56
        assert all(torch.count_nonzero(value) == 0 for key, value in initial.items() if 'lora_B' in key)
        set_trainable(model, 'language')
        model.functional_initialization = dict(origin=origin, full_weight_path=str(ORIGINS[origin].relative_to(ROOT)),
            adapter_count=1, parent_adapter_transferred=False, stacked_adapters=False,
            language_tensors=112, fresh_B_zero=True, full_weight_loading_information=information)
    elif adapter:
        adapter = Path(adapter)
        configuration = read(adapter / 'adapter_config.json')
        assert Path(configuration['base_model_name_or_path']).resolve() == ORIGINS[origin].resolve(), 'Wrong full-weight origin for adapter'
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
        verify_adapter(model, adapter)
    else:
        model = base
    if not train:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
    return model

"""Experimental decision schema and matched language/visual-merger LoRA runtime.

Legacy models, their prompts and their editor remain unchanged. Both new training
routes start with the same exact language adapter; the added merger B matrices
start at zero. This module does not select a checkpoint or admit training data.
"""
from pathlib import Path

import torch

from functional_v1.common import BASE
from functional_v1.runtime import processor, encode, generate, verify_adapter
from .outcomes import input_row, target_text, parse


LANGUAGE_PATTERN = r'.*language_model\.layers\.\d+\.self_attn\.(?:q_proj|v_proj)$'
MERGER_PATTERN = r'.*(?:language_model\.layers\.\d+\.self_attn\.(?:q_proj|v_proj)|visual\.merger\.mlp\.(?:0|2))$'
ROUTES = ('language', 'language_merger')


def canonical(key):
    return key.replace('.model.language_model.', '.model.')


def build_batch(p, case, language, decision):
    row = input_row(case, language)
    batch = encode(p, row, target_text(decision))
    prefix = encode(p, row)['input_ids']
    count = prefix.shape[1]
    assert torch.equal(batch['input_ids'][:, :count], prefix), 'Assistant label prefix mismatch'
    if batch['input_ids'].shape[1] > 2048:
        raise ValueError('Sequence too long; label truncation is forbidden')
    batch['labels'] = batch['input_ids'].clone()
    batch['labels'][:, :count] = -100
    assert int(batch['labels'].ne(-100).sum()) > 4
    return batch


def create_adapter(base, route):
    from peft import LoraConfig, get_peft_model
    assert route in ROUTES
    return get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0., bias='none', task_type='CAUSAL_LM',
        target_modules=LANGUAGE_PATTERN if route == 'language' else MERGER_PATTERN))


def transfer_parent(model, parent, route):
    """Copy all 112 parent language tensors exactly into one new adapter."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    assert route in ROUTES and set(model.peft_config) == {'default'}
    saved = load_file(str(Path(parent) / 'adapter_model.safetensors'), device='cpu')
    expected = {canonical(k): v for k, v in saved.items()}
    assert len(expected) == len(saved) == 112
    assert all('visual' not in k for k in expected)
    actual = get_peft_model_state_dict(model)
    assert len(actual) == (112 if route == 'language' else 116)
    language_keys = {k for k in actual if 'visual' not in k}
    assert {canonical(k) for k in language_keys} == set(expected)
    added = {k: v.detach().cpu().clone() for k, v in actual.items() if 'visual' in k}
    assert len(added) == (0 if route == 'language' else 4)
    assert all(torch.count_nonzero(v) == 0 for k, v in added.items() if 'lora_B' in k)
    state = {k: expected[canonical(k)].to(dtype=v.dtype) if k in language_keys else v for k, v in actual.items()}
    loaded = set_peft_model_state_dict(model, state)
    assert not loaded.unexpected_keys
    after = get_peft_model_state_dict(model)
    assert all(torch.equal(after[k].detach().cpu().to(expected[canonical(k)].dtype), expected[canonical(k)])
               for k in language_keys), 'Parent language tensors changed during transfer'
    assert all(torch.equal(after[k].detach().cpu(), value) for k, value in added.items())
    return dict(route=route, language_parent_tensors_copied_exactly=112,
                added_visual_merger_tensors=len(added), visual_merger_B_initially_zero=True,
                adapter_count=len(model.peft_config), stacked_adapters=False)


def set_trainable(model, route):
    count = 0
    names = []
    for name, value in model.named_parameters():
        active = 'lora_' in name
        value.requires_grad_(active)
        if active:
            assert ('.q_proj.' in name or '.v_proj.' in name) if 'visual' not in name else route == 'language_merger' and '.merger.mlp.' in name
            value.data = value.data.float()
            count += 1
            names.append(name)
    assert count == (112 if route == 'language' else 116)
    model.config.use_cache = False
    model.train()
    return names


def load_model(adapter=None, train=False, route=None, parent=None):
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel, prepare_model_for_kbit_training
    assert not train or adapter is None, 'Training initializes one adapter from the parent, never stacks adapters'
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    if train:
        assert route in ROUTES and parent is not None
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={'use_reentrant': False})
        model = create_adapter(base, route)
        model.functional_parent_transfer = transfer_parent(model, parent, route)
        set_trainable(model, route)
    elif adapter:
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
        verify_adapter(model, adapter)
    else:
        model = base
    if not train:
        for weight in model.parameters():
            weight.requires_grad_(False)
        model.eval()
    return model


def predict(model, p, case, language):
    response = generate(model, p, encode(p, input_row(case, language)), max_tokens=256)
    return response, parse(response)

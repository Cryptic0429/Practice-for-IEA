"""Continue exactly one existing 116-tensor adapter with the new frozen tool manual and unchanged R11 editor."""
from pathlib import Path
import torch
from functional_v1.common import ROOT, BASE, read, sha
from functional_v1.runtime import processor, encode, generate, verify_adapter
from functional_v7.runtime import set_trainable
from functional_v9.runtime import verify_origin as verify_original_origin
from .outcomes import input_row, target_text, parse

EXP = ROOT/'experiments/functional-v12-20261106'
DATA = ROOT/'data/training/functional-v12-20261106'
PARENT = ROOT/'experiments/functional-v11-20261106/runs/language_merger-seed-20261123/final'
PARENT_SHA = '12243ea5f08ab9e2e75b082fd6e6a1e6e8b1efbd7eb4e102be3563ba084c4a56'
ROUTES = ('language_merger',)


def verify_origin():
    identity = verify_original_origin('base')
    assert sha(PARENT/'adapter_model.safetensors') == PARENT_SHA
    config = read(PARENT/'adapter_config.json')
    assert Path(config['base_model_name_or_path']).resolve() == BASE.resolve()
    assert config['r'] == 8 and config['lora_alpha'] == 16 and config['lora_dropout'] == 0
    return identity


def transfer_all(model):
    """Copy language and already-trained visual merger; no new zero-B modules."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from functional_v7.runtime import canonical
    saved = load_file(str(PARENT/'adapter_model.safetensors'), device='cpu')
    expected = {canonical(k): v for k,v in saved.items()}
    actual = get_peft_model_state_dict(model)
    assert len(saved) == len(expected) == len(actual) == 116
    assert {canonical(k) for k in actual} == set(expected)
    assert sum('visual' in k for k in expected) == 4
    result = set_peft_model_state_dict(model, {k:expected[canonical(k)].to(v.dtype) for k,v in actual.items()})
    assert not result.unexpected_keys
    verify_adapter(model, PARENT)
    assert set(model.peft_config) == {'default'}
    return dict(language_parent_tensors_copied_exactly=112, visual_parent_tensors_copied_exactly=4,
                parent_tensors_copied_exactly=116, adapter_count=1, stacked_adapters=False)


def build_batch(p, case, language, decision):
    row = input_row(case, language)
    batch = encode(p, row, target_text(decision)); prefix = encode(p, row)['input_ids']
    count = prefix.shape[1]
    assert torch.equal(batch['input_ids'][:, :count], prefix)
    assert batch['input_ids'].shape[1] <= 2048, 'Truncation forbidden'
    batch['labels'] = batch['input_ids'].clone(); batch['labels'][:, :count] = -100
    assert int(batch['labels'].ne(-100).sum()) > 4
    return batch


def load_model(route, adapter=None, train=False, seed=None):
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, set_seed
    from peft import PeftModel, prepare_model_for_kbit_training
    from functional_v7.runtime import create_adapter
    assert route in ROUTES and (not train or (adapter is None and seed is not None))
    verify_origin()
    base, information = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'':0}, output_loading_info=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    assert not any(information.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')), information
    if train:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={'use_reentrant':False})
        set_seed(seed)
        model = create_adapter(base, route)
        transfer = transfer_all(model)
        set_trainable(model, route)
        model.functional_initialization = dict(origin='base', parent_path=str(PARENT.relative_to(ROOT)),
            parent_sha256=PARENT_SHA, full_weight_loading_information=information, **transfer)
    else:
        assert adapter is not None
        assert Path(read(Path(adapter)/'adapter_config.json')['base_model_name_or_path']).resolve() == BASE.resolve()
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
        verify_adapter(model, adapter)
        for weight in model.parameters(): weight.requires_grad_(False)
        model.eval()
    return model


def predict(model, p, case, language):
    response = generate(model, p, encode(p, input_row(case, language)), max_tokens=256)
    return response, parse(response)

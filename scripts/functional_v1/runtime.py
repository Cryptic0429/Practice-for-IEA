"""Shared, exact model path for curation, training, offline evaluation and demo."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from .common import BASE, PARENT, input_row, system_prompt, validate, parse, sha


def processor():
    from transformers import AutoProcessor
    p = AutoProcessor.from_pretrained(BASE, use_fast=True)
    p.tokenizer.padding_side = 'left'
    p.image_processor.max_pixels = 262144
    return p


def image_input(path):
    from PIL import Image, ImageOps
    import math
    with Image.open(path) as im:
        result = ImageOps.exif_transpose(im).convert('RGB')
        w, h = result.size
        if w*h > 262144:
            factor = math.sqrt(262144/(w*h))
            result = result.resize((max(28, int(w*factor)), max(28, int(h*factor))))
        if min(result.size) < 28:
            result = result.resize((max(28, result.width), max(28, result.height)))
        return result


def messages(row):
    result = []
    for item in row['conversations']:
        if item['role'] not in ('system', 'user'):
            continue
        content = item['content']
        if item['role'] == 'user':
            before, after = content.split('<image>')
            content = ([{'type': 'text', 'text': before}] if before else []) + [
                {'type': 'image'}, {'type': 'text', 'text': after}]
        result.append({'role': item['role'], 'content': content})
    assert [m['role'] for m in result] == ['system', 'user']
    return result


def encode(p, row, target=None):
    msg = messages(row)
    if target is not None:
        msg += [{'role': 'assistant', 'content': target}]
    return p(text=[p.apply_chat_template(msg, tokenize=False, add_generation_prompt=target is None)],
             images=[image_input(row['images'][0])], return_tensors='pt', add_special_tokens=False)


def build_batch(p, case, language, parameters):
    row = input_row(case, language)
    target = '```json\n' + json.dumps(validate(parameters), indent=2) + '\n```'
    batch = encode(p, row, target)
    prefix = encode(p, row)['input_ids']
    count = prefix.shape[1]
    if not torch.equal(batch['input_ids'][:, :count], prefix):
        raise AssertionError('Assistant label prefix mismatch; do not silently shift the loss mask')
    if batch['input_ids'].shape[1] > 2048:
        raise ValueError('Sequence too long; target truncation is forbidden')
    batch['labels'] = batch['input_ids'].clone()
    batch['labels'][:, :count] = -100
    assert int((batch['labels'] != -100).sum()) > 4
    return batch


def load_model(adapter=None, train=False):
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    if train:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                              gradient_checkpointing_kwargs={'use_reentrant': False})
    if adapter:
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=train)
        verify_adapter(model, adapter)
    elif train:
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0., bias='none',
                              target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    else:
        model = base
    if train:
        count = 0
        for name, weight in model.named_parameters():
            active = 'lora_' in name
            weight.requires_grad_(active)
            if active:
                assert 'visual' not in name and ('.q_proj.' in name or '.v_proj.' in name)
                weight.data = weight.data.float()
                count += 1
        assert count == 112
        model.config.use_cache = False
        model.train()
    else:
        for weight in model.parameters():
            weight.requires_grad_(False)
        model.eval()
    return model


def verify_adapter(model, directory):
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    canonical = lambda key: key.replace('.model.language_model.', '.model.')
    expected = {canonical(k): v for k, v in load_file(str(Path(directory)/'adapter_model.safetensors')).items()}
    actual = {canonical(k): v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}
    assert set(actual) == set(expected), 'Adapter module names do not match the saved model'
    assert all(torch.equal(actual[k].to(expected[k].dtype), expected[k]) for k in actual), 'Adapter values changed while loading'


def generate(model, p, inputs, max_tokens=256):
    moved = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        result = model.generate(**moved, do_sample=False, max_new_tokens=max_tokens,
            repetition_penalty=1., pad_token_id=p.tokenizer.pad_token_id,
            eos_token_id=p.tokenizer.eos_token_id, use_cache=True, disable_compile=True)
    tokens = result[0, moved['input_ids'].shape[1]:].tolist()
    if len(tokens) >= max_tokens and p.tokenizer.eos_token_id not in tokens:
        raise ValueError('Generation hit its length limit; do not repair a truncated response')
    return p.decode(tokens, skip_special_tokens=True)


def predict(model, p, case, language):
    response = generate(model, p, encode(p, input_row(case, language)), max_tokens=256)
    return response, parse(response)


REVIEW_SYSTEM = (
    'You are a careful independent photograph editing inspector. The first image is BEFORE '
    'and the second is AFTER. Judge the visible change against the supplied request. '
    'Do not assume any edit happened merely because an output exists. '
    'A visible but opposite, insufficient, or excessive change is not completed. '
    'Report clipped sky, lost dark detail, unwanted colour casts and unrelated changes. '
    'For a brown-tone request, green-dominated foliage with a yellow cast is not a brown transformation. '
    'No reference photograph, editing parameters, model identity or pixel distance is provided. '
    'Return only a JSON object with these fields: '
    'completion: completed|partial|failed|uncertain; '
    'quality: acceptable|minor_issue|major_issue|uncertain; '
    'strength: appropriate|insufficient|excessive|uncertain; '
    'evidence: one concise sentence describing the visible change. '
    'Use uncertain when the change or quality cannot be confidently assessed.'
)


def judge(model, p, case, output):
    from .common import image_path
    request = (case['instructions']['en'] + '\nChinese equivalent: ' + case['instructions']['zh'] +
               '\nRequired change: ' + case['expected_change'] + '\nPreserve: ' + case['must_preserve'])
    msg = [{'role': 'system', 'content': REVIEW_SYSTEM},
           {'role': 'user', 'content': [{'type': 'text', 'text': 'BEFORE:'}, {'type': 'image'},
             {'type': 'text', 'text': 'AFTER:'}, {'type': 'image'}, {'type': 'text', 'text': request}]}]
    batch = p(text=[p.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)],
              images=[image_input(image_path(case)), image_input(output)],
              return_tensors='pt', add_special_tokens=False)
    response = generate(model, p, batch, max_tokens=180)
    text = response.strip()
    if text.startswith('```json') and text.endswith('```'):
        text = text[7:-3].strip()
    result = json.loads(text)
    allowed = {'completion': {'completed','partial','failed','uncertain'},
               'quality': {'acceptable','minor_issue','major_issue','uncertain'},
               'strength': {'appropriate','insufficient','excessive','uncertain'}}
    for field, values in allowed.items():
        if result.get(field) not in values:
            raise ValueError('Invalid visual-review field: ' + field)
    if not isinstance(result.get('evidence'), str) or not result['evidence'].strip():
        raise ValueError('Visual evidence is missing')
    return result

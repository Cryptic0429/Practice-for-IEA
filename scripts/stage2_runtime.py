"""Shared Stage 2 model/input primitives. No training on import."""
import contextlib
import hashlib
import math
import types
import weakref
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel, prepare_model_for_kbit_training, get_peft_model_state_dict
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

ROOT = Path('/mnt/sdc/zhc/IEA')
BASE = ROOT/'models/Qwen2.5-VL-7B-Instruct'
ADAPTER = ROOT/'outputs/stage1-A-image-edit-3k-v1'


def tensor_digest(tensors):
    h = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def adapter_state(model, name='default'):
    return {k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model, adapter_name=name).items()}


def image_input(path):
    # Matches LLaMA-Factory BasePlugin preprocessing; Qwen processor then smart-resizes.
    with Image.open(path) as source:
        image = source.copy()
    for limit, larger in [(262144, True), (1024, False)]:
        area = image.width * image.height
        if (area > limit) if larger else (area < limit):
            factor = math.sqrt(limit/area)
            image = image.resize((int(image.width*factor), int(image.height*factor)))
    image = image.convert('RGB')
    if min(image.size) < 28:
        image = image.resize((max(image.width,28),max(image.height,28)))
    if image.width/image.height > 200:
        image = image.resize((image.height*180,image.height))
    if image.height/image.width > 200:
        image = image.resize((image.width,image.width*180))
    return image


def prompt(row):
    messages = []
    for m in row['conversations']:
        if m['role'] not in ('system', 'user'):
            continue
        content = m['content']
        if m['role'] == 'user':
            assert content.count('<image>') == 1
            before, after = content.split('<image>')
            parts = ([dict(type='text', text=before)] if before else [])
            parts += [dict(type='image')]
            if after:
                parts.append(dict(type='text', text=after))
            content = parts
        messages.append(dict(role=m['role'], content=content))
    assert [m['role'] for m in messages] == ['system','user']
    assert row['images'] == [row['extra_info']['origin_image_path']]
    return messages


def processor():
    p = AutoProcessor.from_pretrained(BASE, use_fast=True)
    p.tokenizer.padding_side = 'left'
    return p


def encoded(p, row):
    return p(text=[p.apply_chat_template(prompt(row), tokenize=False, add_generation_prompt=True)],
             images=[image_input(row['images'][0])], return_tensors='pt', add_special_tokens=False)


def load_model(adapter=ADAPTER, trainable=True):
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'':0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    if trainable:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                              gradient_checkpointing_kwargs={'use_reentrant':False})
    model = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
    from safetensors.torch import load_file
    saved = load_file(str(Path(adapter)/'adapter_model.safetensors'))
    # Transformers 4.52+ renamed the language backbone; verify every loaded value.
    canonical = lambda k: k.replace('.model.language_model.', '.model.')
    loaded = {canonical(k):v for k,v in adapter_state(model).items()}
    expected = {canonical(k):v for k,v in saved.items()}
    assert loaded.keys() == expected.keys(), 'Adapter key mapping differs'
    assert all(torch.equal(loaded[k],v) for k,v in expected.items()), 'A adapter load changed tensors'
    if trainable:
        model.load_adapter(ADAPTER, adapter_name='reference', is_trainable=False)
        # The second load must use the already verified current backbone key names.
        # PEFT's first from_pretrained conversion and later load_adapter can differ.
        from peft import set_peft_model_state_dict
        origin = {canonical(k):v for k,v in load_file(str(ADAPTER/'adapter_model.safetensors')).items()}
        a = {k:origin[canonical(k)] for k in adapter_state(model)}
        set_peft_model_state_dict(model, a, adapter_name='reference')
        b=adapter_state(model,'reference')
        assert a.keys()==b.keys(), (list(a)[:2],list(b)[:2])
        assert all(torch.equal(a[k],b[k]) for k in a), 'Frozen A clone differs'
        model.set_adapter('default')
        for n,p in model.named_parameters():
            p.requires_grad_('.default.' in n and 'lora_' in n)
        assert_trainable(model)
    return model


def assert_trainable(model):
    names = [n for n,p in model.named_parameters() if p.requires_grad]
    assert len(names) == 112, (len(names), names)
    assert all('lora_' in n and '.default.' in n and '.self_attn.' in n and
               ('q_proj.' in n or 'v_proj.' in n) and 'visual' not in n for n in names)
    return names


class FrozenReference(torch.nn.Module):
    """Shares quantized base; reference adapter is never included in optimizer."""
    def __init__(self, policy):
        super().__init__()
        object.__setattr__(self, '_policy_ref', weakref.ref(policy))

    def forward(self, **kwargs):
        model = self._policy_ref()
        was_training = model.training
        try:
            model.set_adapter('reference')
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            with torch.no_grad():
                return model(**kwargs)
        finally:
            model.set_adapter('default')
            model.train(was_training)
            for n,p in model.named_parameters():
                p.requires_grad_('.default.' in n and 'lora_' in n)


def chunk_generation(model, concurrency=1):
    """TRL retains16-candidate grouping; resource-only HF generation chunking."""
    assert concurrency in [1,2,4]
    original = model.generate

    def generate(self, **kwargs):
        ids = kwargs['input_ids']
        grid = kwargs['image_grid_thw']
        assert grid.shape[0] == ids.shape[0], 'Exactly one original image per candidate'
        lengths = grid.prod(-1).tolist()
        offset, outputs = 0, []
        for i in range(0,len(lengths),concurrency):
            end=min(i+concurrency,len(lengths))
            count=sum(lengths[i:end])
            one = dict(kwargs)
            for key,value in kwargs.items():
                if key == 'pixel_values':
                    one[key] = value[offset:offset+count]
                elif torch.is_tensor(value) and value.shape[0] == ids.shape[0]:
                    one[key] = value[i:end]
            offset += count
            # Transformers 4.56.2 Qwen2.5-VL incremental RoPE combines a
            # padding-excluded text position with a delta that includes padding.
            # Generate each same-prompt candidate group without its common left
            # padding, then restore that prefix for TRL's fixed prompt slicing.
            mask = one.get('attention_mask')
            trim = 0
            if mask is not None:
                starts = mask.long().argmax(-1)
                assert torch.all(starts == starts[0]), 'Chunk must contain equally padded prompts'
                trim = int(starts[0])
                if trim:
                    one['input_ids'] = one['input_ids'][:,trim:]
                    one['attention_mask'] = mask[:,trim:]
                assert torch.all(one['attention_mask'] == 1), 'Generation chunk still contains masked tokens'
            generated = original(**one)
            if trim:
                generated = torch.cat([ids[i:end,:trim], generated], dim=1)
            assert torch.equal(generated[:,:ids.shape[1]],ids[i:end]), 'Generation changed the prompt prefix'
            outputs.extend(generated)
        config = kwargs.get('generation_config')
        pad = config.pad_token_id if config is not None else kwargs['pad_token_id']
        return torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True, padding_value=pad)

    model.generate = types.MethodType(generate, model)
    return original

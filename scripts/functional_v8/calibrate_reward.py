"""Frozen external reward calibration; no policy training, editor change or repair."""
import argparse
import ast
import hashlib
import json
import math
import random
import re
import signal

from functional_v1.common import ROOT, BASE, GPUJob, read, write, sha, stamp
from .prepare_calibration import EXP, DATA, REWARD_SHA


def prompts():
    path = ROOT / 'third_party/EditScore/editscore/vie_prompts.py'
    constants = {}
    for node in ast.parse(path.read_text()).body:
        assert isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        constants[node.targets[0].id] = ast.literal_eval(node.value)
    context = constants['_context_no_delimit_reasoning_first']
    sc = '\n'.join([context, constants['_prompts_0shot_two_image_edit_rule'],
                    constants['_prompts_0shot_tie_rule_SC'].replace('10', '25')])
    pq = '\n'.join([context, constants['_prompts_0shot_rule_PQ'].replace('10', '25')])
    return sc, pq


def tasks():
    dataset = read(DATA / 'calibration-candidates.json')
    sc, pq = prompts()
    result = []
    for case in dataset['direction']:
        for language in ('zh', 'en'):
            for role, key in [('admitted', 'good_image'), ('identity', 'input_image'), ('opposite', 'bad_image')]:
                result.append(dict(id=f'{case["id"]}-{language}-{role}', case_id=case['id'], axis='SC',
                    language=language, role=role, images=[case['input_image'], case[key]],
                    prompt=sc.replace('<instruction>', case['instructions'][language])))
    for case in dataset['quality']:
        for role, key in [('admitted', 'good_image'), ('damaged', 'bad_image')]:
            result.append(dict(id=f'{case["id"]}-{role}', case_id=case['id'], axis='PQ', language=None,
                               role=role, images=[case[key]], prompt=pq))
    return sorted(result, key=lambda row: hashlib.sha256(row['id'].encode()).hexdigest())


def score_json(response):
    value = response.strip()
    if value.startswith('```'):
        match = re.fullmatch(r'```(?:json)?\s*(\{.*\})\s*```', value, re.S)
        if not match:
            raise ValueError('Invalid JSON fence; no repair or default score')
        value = match.group(1)
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key; no last-value fallback')
            result[key] = value
        return result
    decoded = json.loads(value, object_pairs_hook=unique_keys)
    if not isinstance(decoded, dict) or set(decoded) != {'reasoning', 'score'} or not isinstance(decoded['reasoning'], str):
        raise ValueError('Expected exactly reasoning and score')
    scores = decoded['score']
    if not isinstance(scores, list) or len(scores) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 25 for v in scores):
        raise ValueError('Expected two finite scores in [0,25]')
    return decoded


def encode(processor, task):
    from qwen_vl_utils import process_vision_info
    images = [{'type': 'image', 'image': str(ROOT / path), 'max_pixels': 262144} for path in task['images']]
    messages = [{'role': 'user', 'content': images + [{'type': 'text', 'text': task['prompt']}]}]
    # Exact official Qwen2.5 manual template including <img1>/<img2> labels.
    text = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    text += ''.join(f'<img{i}>: <|vision_start|><|image_pad|><|vision_end|>' for i in range(1, len(images) + 1))
    text += task['prompt'] + '<|im_end|>\n<|im_start|>assistant\n'
    image_inputs, video_inputs = process_vision_info(messages)
    result = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt')
    assert result['input_ids'].shape[1] <= 4096
    return result


def main(route, gpu):
    import torch
    import numpy as np
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    from functional_v1.runtime import verify_adapter
    protocol = read(EXP / 'calibration-protocol.json')
    for path, digest in protocol['code_hashes'].items():
        assert sha(ROOT / path) == digest
    assert sha(DATA / 'calibration-candidates.json') == protocol['dataset_sha256']
    assert sha(EXP / 'calibration-direct-observations.jsonl') == protocol['direct_review_sha256']
    folder = EXP / 'reward-calibration' / route
    assert not folder.exists(), 'Preserve previous run and raw responses'
    folder.mkdir(parents=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit('Owned reward calibration stopped')))
    torch.set_num_threads(4)
    adapter = ROOT / 'models/external/EditScore-7B'
    assert sha(adapter / 'adapter_model.safetensors') == REWARD_SHA
    rows = tasks(); assert len(rows) == 76
    with GPUJob(gpu, 'round8-frozen-reward-calibration-' + route, max_seconds=1800) as budget:
        with budget.ledger() as ledger:
            next(j for j in ledger['jobs'] if j['id'] == budget.id).update(
                editor='none_frozen_outcome_scoring_only', runtime='functional_v8.calibrate_reward',
                reward_model=route, policy_training=False)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                         attn_implementation='sdpa', device_map={'': 0})
        if route == 'editscore':
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
            verify_adapter(model, adapter)
            model = model.merge_and_unload()
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        processor = AutoProcessor.from_pretrained(BASE, max_pixels=262144)
        write(folder / 'identity.json', dict(at=stamp(), route=route, model_origin='External pretrained frozen reward adapter' if route == 'editscore' else 'Original Qwen baseline under same reward prompt',
            adapter_sha256=REWARD_SHA if route == 'editscore' else None, precision='BF16 full base, no NF4',
            adapter_merged=route == 'editscore', max_pixels_each_image=262144,
            sampling=dict(seed=20261114, temperature=.7, top_p=.9, top_k=20, max_new_tokens=512, passes=1),
            protocol_sha256=sha(EXP / 'calibration-protocol.json'), own_policy_training=False))
        done = 0
        for task in rows:
            budget.tick()
            record = {k: task[k] for k in ('id', 'case_id', 'axis', 'language', 'role', 'images')}
            record.update(at=stamp(), image_sha256=[sha(ROOT / p) for p in task['images']],
                          prompt_sha256=hashlib.sha256(task['prompt'].encode()).hexdigest())
            try:
                batch = encode(processor, task)
                record['input_tokens'] = batch['input_ids'].shape[1]
                moved = {key: value.to('cuda') for key, value in batch.items()}
                random.seed(20261114); np.random.seed(20261114); torch.manual_seed(20261114); torch.cuda.manual_seed_all(20261114)
                with torch.inference_mode():
                    output = model.generate(**moved, max_new_tokens=512, do_sample=True, temperature=.7,
                                            top_p=.9, top_k=20, use_cache=True, disable_compile=True,
                                            pad_token_id=processor.tokenizer.pad_token_id)
                tokens = output[0, moved['input_ids'].shape[1]:]
                response = processor.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                record['raw_response'] = response
                if len(tokens) >= 512 and processor.tokenizer.eos_token_id not in tokens:
                    raise ValueError('Length-limited response; no repair')
                record.update(status='scored', **score_json(response))
            except Exception as error:
                record.update(status='invalid', error=f'{type(error).__name__}: {error}')
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    with (folder / 'predictions.jsonl').open('a') as file:
                        file.write(json.dumps(record, ensure_ascii=False) + '\n')
                    raise
            with (folder / 'predictions.jsonl').open('a') as file:
                file.write(json.dumps(record, ensure_ascii=False) + '\n')
            done += 1
            write(folder / 'status.json', dict(at=stamp(), status='running', completed=done, total=76))
        write(folder / 'status.json', dict(at=stamp(), status='completed', completed=done, total=76,
                                          functional_improvement_claimed=False))
    print(route, '76 frozen calibration outputs recorded; no policy training or automatic success claim')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--route', required=True, choices=['base', 'editscore'])
    parser.add_argument('--gpu', required=True, choices=['0', '2', '3'])
    args = parser.parse_args()
    main(args.route, args.gpu)

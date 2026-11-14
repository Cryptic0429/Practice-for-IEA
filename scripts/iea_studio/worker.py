"""Disposable inference process: one frozen base, two mutually exclusive adapters."""
import argparse
import os
import threading
import time
import traceback
from pathlib import Path
from .common import ROOT, MODELS, read, write, sha, stamp

# PEFT's state export filters by substring. Names must not contain one another.
ADAPTER_NAMES = {'r1': 'first_candidate', 'r15': 'parent_candidate'}


class Engine:
    def __init__(self, device='cpu', threads=20):
        import torch
        from peft import PeftModel, get_peft_model_state_dict
        from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
        from safetensors.torch import load_file
        from functional_v1.common import BASE
        from functional_v1.runtime import processor
        from functional_v9.runtime import verify_origin
        from functional_v21.head import load_head
        from functional_v21.common import protocol, EXP, PARENT_SHA

        self.device = device
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        verify_origin('base')
        self.identities = {}
        for name, spec in MODELS.items():
            assert sha(ROOT / spec['adapter'] / 'adapter_model.safetensors') == spec['sha256'], name
            self.identities[name] = dict(spec, device=device,
                precision='NF4 double quantization / BF16' if device == 'cuda' else 'FP32',
                max_image_pixels=262144, max_new_tokens=256, do_sample=False)
        head_spec = MODELS['r21']
        assert sha(ROOT / head_spec['controller']) == head_spec['controller_sha256']
        protocol()
        self.head, self.head_weights = load_head(ROOT / head_spec['controller'])
        assert self.head_weights['parent_sha256'] == PARENT_SHA
        assert self.head_weights['protocol_sha256'] == sha(EXP / 'training-protocol.json')
        kwargs = dict(local_files_only=True, attn_implementation='sdpa', output_loading_info=True)
        if device == 'cuda':
            kwargs.update(torch_dtype=torch.bfloat16, device_map={'': 0},
                quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
        else:
            kwargs.update(torch_dtype=torch.float32, device_map={'': 'cpu'})
        base, loading = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE, **kwargs)
        assert not any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), loading
        self.model = PeftModel.from_pretrained(base, str(ROOT / MODELS['r1']['adapter']),
                                             adapter_name=ADAPTER_NAMES['r1'], is_trainable=False)
        for name in ('r15',):
            self.model.load_adapter(str(ROOT / MODELS[name]['adapter']), adapter_name=ADAPTER_NAMES[name], is_trainable=False)
        canonical = lambda k: k.replace('.model.language_model.', '.model.')
        for name in ('r1', 'r15'):
            expected = {canonical(k): v for k, v in load_file(str(ROOT / MODELS[name]['adapter'] / 'adapter_model.safetensors')).items()}
            actual = {canonical(k): v.detach().cpu() for k, v in get_peft_model_state_dict(self.model, adapter_name=ADAPTER_NAMES[name]).items()}
            assert set(actual) == set(expected), (name, set(actual) ^ set(expected))
            assert all(torch.equal(actual[k].to(v.dtype), v) for k, v in expected.items()), name
        self.model.eval()
        for weight in self.model.parameters():
            weight.requires_grad_(False)
        self.processor = processor()
        self.load_seconds = None

    def generate(self, model_id, image, instruction, need_features=False):
        import torch
        import contextlib
        from functional_v1.runtime import encode
        from functional_v1.common import input_row as r1_row, parse as r1_parse
        from functional_v15.outcomes import input_row, parse
        from functional_v19.photometry import features
        name = 'r15' if model_id == 'r21' else model_id
        self.model.set_adapter(ADAPTER_NAMES[name])
        self.model.eval()
        assert self.model.active_adapters == [ADAPTER_NAMES[name]], self.model.active_adapters
        case = dict(image=str(image), instructions={'original': instruction})
        inputs = encode(self.processor, (r1_row if name == 'r1' else input_row)(case, 'original'))
        encoded = {k: v.to(self.device) for k, v in inputs.items()}
        prefix = encoded['input_ids'].shape[1]
        p = self.processor
        context = torch.autocast('cuda', dtype=torch.bfloat16) if self.device == 'cuda' else contextlib.nullcontext()
        with torch.inference_mode(), context:
            generated = self.model.generate(**encoded, do_sample=False, max_new_tokens=256,
                repetition_penalty=1., pad_token_id=p.tokenizer.pad_token_id,
                eos_token_id=p.tokenizer.eos_token_id, use_cache=True, disable_compile=True)
            tokens = generated[0, prefix:].tolist()
            if len(tokens) >= 256 and p.tokenizer.eos_token_id not in tokens:
                raise ValueError('输出达到长度上限，未执行不完整参数')
            raw = p.decode(tokens, skip_special_tokens=True)
            decision = (r1_parse if name == 'r1' else parse)(raw)
            observed = None
            if need_features and decision.get('action') == 'edit':
                hidden = self.model.get_base_model().model(**encoded, use_cache=False, return_dict=True).last_hidden_state
                observed = dict(last=hidden[0, -1].float().cpu(), stats=torch.from_numpy(features(image)))
        return raw, decision, observed

    def controlled(self, image, decision, observed):
        import torch
        from functional_v21.head import predict_cached
        from functional_v21.texture import features
        if decision['action'] != 'edit':
            return decision
        _, result = predict_cached(self.head, self.head_weights, observed,
            torch.from_numpy(features(image)), decision, 'magnitude-quality')
        return result

    def render(self, name, image, decision, output):
        from functional_v1.editor import load_rgb, save, edit
        from functional_v15.outcomes import execute
        rgb = load_rgb(image)
        if name == 'r1':
            pixels = edit(rgb, decision)
            rendered = dict(status='edited' if any(decision.values()) else 'unchanged',
                            parameters=decision, result=pixels)
        else:
            rendered = execute(rgb, decision)
        result = dict(status=rendered['status'], parameters=rendered.get('parameters'),
                      reason=rendered.get('reason_code'), after=None)
        if rendered['status'] != 'declined':
            save(output, rendered['result'])
            result.update(after=f'/files/{output.parent.name}/{output.name}', output_sha256=sha(output))
        return result


def run(engine, folder):
    job_file = folder / 'job.json'
    job = read(job_file)
    image = folder / 'input.png'
    assert sha(image) == job['input_sha256']
    job.update(status='running', started_at=stamp(), identity=engine.identities,
               load_seconds=engine.load_seconds)
    write(job_file, job)
    parent = {}
    # Run R15 first so the core R15+R21 pipeline finishes before the R1 baseline.
    for name in ('r15', 'r21', 'r1'):
        for language in ('zh', 'en'):
            row = next(r for r in job['results'] if r['model'] == name and r['language'] == language)
            row.update(status='running', started_at=stamp())
            job['message'] = f"正在处理 {MODELS[name]['name']} · {'中文' if language == 'zh' else 'English'}"
            write(job_file, job)
            started = time.monotonic()
            try:
                if name == 'r21':
                    raw, decision, features, parent_seconds = parent[language]
                    decision = engine.controlled(image, decision, features)
                    row.update(parent_raw=raw, shared_parent='r15', parent_seconds=parent_seconds)
                else:
                    raw, decision, features = engine.generate(name, image, job['instructions'][language], name == 'r15')
                    row['raw'] = raw
                    if name == 'r15':
                        parent[language] = (raw, decision, features, time.monotonic() - started)
                row.update(engine.render(name, image, decision, folder / f'{name}-{language}.png'))
                row['decision'] = decision
            except Exception as error:
                traceback.print_exc()
                (folder / f'{name}-{language}.png').unlink(missing_ok=True)
                row.update(status='failed', after=None, error=f'{type(error).__name__}: {error}')
            row['seconds'] = round(time.monotonic() - started, 2)
            row['completed_at'] = stamp()
            write(job_file, job)
    job.update(status='completed', completed_at=stamp(), message='核心方案与对照基线的双语对比已完成')
    write(job_file, job)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=20)
    parser.add_argument('--max-seconds', type=float, default=0)
    parser.add_argument('jobs', nargs='+')
    args = parser.parse_args()
    if args.device == 'cuda':
        if args.max_seconds <= 0:
            parser.error('GPU inference requires an explicit remaining time limit')
        # A separate watchdog also enforces the cap if the web server disappears.
        watchdog = threading.Timer(args.max_seconds, lambda: os._exit(124))
        watchdog.daemon = True
        watchdog.start()
    started = time.monotonic()
    engine = Engine(args.device, args.threads)
    engine.load_seconds = round(time.monotonic() - started, 2)
    for folder in args.jobs:
        run(engine, Path(folder))
    if args.device == 'cuda':
        watchdog.cancel()


if __name__ == '__main__':
    main()

"""Loopback-only CPU trial of the selected R21 checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import threading
import time
import traceback
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError

from functional_v1.common import BASE, ROOT, append, read, sha, stamp, write
from functional_v1.catalog import resolve

MODEL_ID = 'functional-v21-quality-20261205'
RUN = ROOT / 'experiments/functional-final-delivery-20261107'
DEMO = RUN / 'trial'
REQUESTS = DEMO / 'requests'
ASSETS = Path(__file__).parent
app = Flask(__name__, template_folder=str(ASSETS),
            static_folder=str(ROOT / 'scripts/functional_v1/web/static'))
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
engine = None
lock = threading.Lock()


class Engine:
    def __init__(self, threads):
        import torch
        from peft import PeftModel
        from transformers import Qwen2_5_VLForConditionalGeneration
        from functional_v1.runtime import processor, verify_adapter
        from functional_v9.runtime import verify_origin
        from functional_v15.outcomes import EDITOR_VERSION, SCHEMA_VERSION
        from functional_v21.common import EXP, PARENT, PARENT_SHA, protocol
        from functional_v21.head import load_head

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        spec = resolve(MODEL_ID)
        proof = protocol()
        verify_origin('base')
        controller = Path(spec['adapter'])
        if sha(controller) != spec['adapter_sha256']:
            raise ValueError('Selected controller hash differs from catalog')
        self.head, self.weights = load_head(controller)
        if self.weights['parent_sha256'] != PARENT_SHA or self.weights['protocol_sha256'] != sha(EXP / 'training-protocol.json'):
            raise ValueError('Selected controller is incompatible with the parent')
        base, loading = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE, torch_dtype=torch.float32, attn_implementation='sdpa',
            device_map={'': 'cpu'}, local_files_only=True, output_loading_info=True)
        if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError(f'Incomplete checkpoint load: {loading}')
        self.backbone = PeftModel.from_pretrained(base, str(PARENT), is_trainable=False).eval()
        verify_adapter(self.backbone, PARENT)
        if any(w.device.type != 'cpu' for w in self.backbone.parameters()):
            raise RuntimeError('Non-CPU parameter in deployment')
        self.processor = processor()
        self.identity = dict(model_id=MODEL_ID, display_name='R21 种子 20261205 · 幅度与画质判断',
            device='cpu', precision='float32, full original Qwen base',
            evaluation_precision='GPU NF4/bfloat16', adapter_sha256=PARENT_SHA,
            controller_sha256=sha(controller), editor=EDITOR_VERSION, schema=SCHEMA_VERSION,
            training_data_sha256=proof['sources']['train']['sha256'],
            editor_sha256=sha(ROOT / 'scripts/functional_v15/editor.py'), training_route=spec['training_route'])

    def infer(self, image, instruction):
        import torch
        from functional_v1.runtime import encode
        from functional_v15.outcomes import execute, input_row, parse
        from functional_v15.editor import load_rgb
        from functional_v19.photometry import features
        from functional_v21.head import predict_cached
        from functional_v21.texture import features as texture_features

        case = dict(image=str(image.relative_to(ROOT)), instructions={'original': instruction})
        p = self.processor
        encoded = encode(p, input_row(case, 'original'))
        prefix = encoded['input_ids'].shape[1]
        with torch.inference_mode():
            generated = self.backbone.generate(**encoded, do_sample=False, max_new_tokens=256,
                repetition_penalty=1., pad_token_id=p.tokenizer.pad_token_id,
                eos_token_id=p.tokenizer.eos_token_id, use_cache=True, disable_compile=True)
            tokens = generated[0, prefix:].tolist()
            if len(tokens) >= 256 and p.tokenizer.eos_token_id not in tokens:
                raise ValueError('Generation reached its output limit')
            decision = parse(p.decode(tokens, skip_special_tokens=True))
            if decision['action'] == 'edit':
                hidden = self.backbone.get_base_model().model(**encoded, use_cache=False,
                                                               return_dict=True).last_hidden_state
                observed = dict(last=hidden[0, -1].float(), stats=torch.from_numpy(features(image)))
                texture = torch.from_numpy(texture_features(image))
                _, decision = predict_cached(self.head, self.weights, observed, texture, decision,
                                             self.identity['training_route'])
        return decision, execute(load_rgb(image), decision)


@app.get('/')
def index():
    return render_template('trial.html', model=engine.identity)


@app.get('/health')
def health():
    return jsonify(status='ready', identity=engine.identity, gpu_hours=0)


@app.post('/api/edit')
def edit_image():
    instruction = request.form.get('instruction', '').strip()
    if not instruction or len(instruction) > 6000 or 'image' not in request.files:
        return jsonify(status='failed', error='请选择图片并填写 1–6000 字的修图要求。'), 400
    if request.form.get('model_id', MODEL_ID) != MODEL_ID:
        return jsonify(status='failed', error='本入口仅提供已选定的 R21 模型。'), 400
    if not lock.acquire(blocking=False):
        return jsonify(status='busy', error='CPU 正在处理上一张图片，请稍后重试。'), 429
    try:
        return run_edit(instruction)
    finally:
        lock.release()


def run_edit(instruction):
    from functional_v15.editor import save

    with Image.open(request.files['image'].stream) as image:
        if image.width * image.height > 40_000_000:
            return jsonify(status='failed', error='图片超过 4000 万像素。'), 400
        rgb = ImageOps.exif_transpose(image).convert('RGB')
    request_id = uuid.uuid4().hex
    folder = REQUESTS / request_id
    folder.mkdir(parents=True, exist_ok=False)
    original = folder / 'input.png'
    output = folder / 'output.png'
    rgb.save(original, compress_level=1)
    source = 'deployment_smoke' if request.headers.get('X-IEA-Source') == 'deployment-smoke' else 'user_trial'
    record = dict(request_id=request_id, instruction=instruction, model_id=MODEL_ID,
                  source=source, training_use=False, created_at=stamp(), status='submitted')
    write(folder / 'request.json', record)
    result = dict(request_id=request_id, before=f'/files/{request_id}/input.png',
                  after=None, output_sha256=None, parameters=None, identity=engine.identity)
    start = time.monotonic()
    code = 200
    try:
        decision, rendered = engine.infer(original, instruction)
        result.update(decision=decision, execution_status=rendered['status'])
        if rendered['status'] == 'declined':
            reason = rendered['reason_code']
            message = ('模型判断无法可靠完成这项要求，未生成结果图。' if reason == 'unreliable_quality'
                       else '这项要求超出当前工具范围，未生成结果图。')
            result.update(status='declined', reason_code=reason, message=message)
        else:
            save(output, rendered['result'])
            result.update(status='executed', after=f'/files/{request_id}/output.png',
                          output_sha256=sha(output), parameters=rendered['parameters'],
                          message='已按要求保持原图。' if rendered['status'] == 'unchanged'
                          else '已生成结果，请对照原图判断是否符合要求。')
    except Exception as error:
        output.unlink(missing_ok=True)
        traceback.print_exc()
        result.update(status='failed', error=f'编辑未完成。{type(error).__name__}: {error}')
        code = 422
    result['seconds'] = round(time.monotonic() - start, 2)
    write(folder / 'request.json', dict(record, **{k: v for k, v in result.items() if k != 'request_id'},
                                       completed_at=stamp()))
    return jsonify(result), code


@app.errorhandler(UnidentifiedImageError)
@app.errorhandler(Image.DecompressionBombError)
def bad_image(_):
    return jsonify(status='failed', error='图片无法读取，或像素超过限制。'), 400


@app.errorhandler(413)
def upload_limit(_):
    return jsonify(status='failed', error='上传大小超过 20 MB。'), 413


def known_request(request_id):
    return re.fullmatch(r'[0-9a-f]{32}', request_id) and (REQUESTS / request_id / 'request.json').is_file()


@app.get('/files/<request_id>/<filename>')
def download(request_id, filename):
    if filename not in ('input.png', 'output.png') or not known_request(request_id):
        return jsonify(error='not_found'), 404
    return send_from_directory(REQUESTS / request_id, filename)


@app.post('/api/feedback')
def feedback():
    value = request.get_json(silent=True)
    if not isinstance(value, dict):
        return jsonify(error='请选择有效的试用反馈。'), 400
    request_id, verdict = value.get('request_id'), value.get('verdict')
    if not isinstance(request_id, str) or not known_request(request_id) or verdict not in (
        'completed', 'partial', 'failed', 'uncertain'
    ):
        return jsonify(error='请选择有效的试用反馈。'), 400
    record = read(REQUESTS / request_id / 'request.json')
    append(DEMO / 'feedback.jsonl', dict(request_id=request_id, verdict=verdict,
        comment=str(value.get('comment', ''))[:2000], at=stamp(), source=record['source'], training_use=False))
    return jsonify(status='saved', message='反馈已保存。')


def main():
    global engine
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=12347)
    ap.add_argument('--threads', type=int, default=20)
    args = ap.parse_args()
    print('Loading selected R21 checkpoint on CPU', flush=True)
    engine = Engine(args.threads)
    print('R21 checkpoint ready', flush=True)
    app.run(host='127.0.0.1', port=args.port, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()

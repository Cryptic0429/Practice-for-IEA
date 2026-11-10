"""Loopback web app. Uploads and reviews are isolated from training data."""
import argparse
import atexit
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid
import zipfile

from flask import Flask, jsonify, request, render_template, send_file, send_from_directory, abort
from PIL import Image, ImageOps, UnidentifiedImageError
from .common import ROOT, RUN, JOBS, MODELS, SAMPLE_NAMES, samples, initial_results, job_path, read, write, sha, stamp, append

ASSETS = Path(__file__).parent
app = Flask(__name__, template_folder=str(ASSETS / 'templates'), static_folder=str(ASSETS / 'static'))
app.config['MAX_CONTENT_LENGTH'] = 22 * 1024 * 1024
runner = None
submit_lock = threading.Lock()
feedback_lock = threading.Lock()


def public(job):
    result = dict(job, before=f"/files/{job['id']}/input.png")
    path = JOBS / job['id'] / 'feedback.json'
    result['feedback'] = read(path) if path.exists() else {}
    if (result['feedback'].get('source') == 'agent_smoke_test'
            and request.headers.get('X-IEA-Source') != 'agent-smoke'):
        result['feedback'] = {}
    result['finished'] = sum(r['status'] not in ('queued', 'running') for r in job['results'])
    return result


class Runner:
    def __init__(self, python, device='cpu', gpu='0', budget_hours=0):
        self.python, self.device, self.gpu = python, device, gpu
        self.cap = budget_hours * 3600
        self.process = None
        self.current = []
        self.running_since = None
        self.wake = threading.Event()
        self.stopped = False
        self.lock = threading.Lock()
        self.ledger_path = RUN / 'resource-ledger.json'
        ledger = read(self.ledger_path) if self.ledger_path.exists() else {'gpu_seconds': 0, 'runs': []}
        # Account conservatively through the last heartbeat if an earlier server stopped.
        self.used = ledger['gpu_seconds']
        self.ledger = ledger
        for p in JOBS.glob('*/job.json'):
            item = read(p)
            if item['status'] in ('running', 'loading'):
                self.fail(p.parent, '服务曾中断；已完成的图片保留，可重新运行比较。')
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def health(self):
        with self.lock:
            elapsed = time.monotonic() - self.running_since if self.running_since and self.device == 'cuda' else 0
            used = self.used + elapsed
        return dict(device=self.device, gpu=self.gpu if self.device == 'cuda' else None,
                    used_gpu_hours=round(used / 3600, 5), budget_gpu_hours=self.cap / 3600,
                    active_jobs=self.current, ready=not self.stopped,
                    precision='NF4 / BF16' if self.device == 'cuda' else 'FP32',
                    gpu_budget_exhausted=self.device == 'cuda' and used >= self.cap)

    def fail(self, folder, message):
        job = read(folder / 'job.json')
        job.update(status='failed', message=message, completed_at=stamp())
        for row in job['results']:
            if row['status'] in ('queued', 'running'):
                row.update(status='failed', error=message, after=None)
        write(folder / 'job.json', job)

    def close(self):
        self.stopped = True
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.wake.set()

    def loop(self):
        while not self.stopped:
            pending = sorted((p for p in JOBS.glob('*/job.json') if read(p)['status'] == 'queued'),
                             key=lambda p: read(p)['created_at'])
            if not pending:
                self.wake.wait(1)
                self.wake.clear()
                continue
            folders = [p.parent for p in pending[:6]]
            if self.device == 'cuda' and self.used >= self.cap - 10:
                for folder in folders:
                    self.fail(folder, '本次新增 GPU 预算已用完；已保存的结果仍可查看和下载。')
                continue
            gpu_lock = None
            try:
                if self.device == 'cuda':
                    import fcntl
                    if self.gpu not in ('0', '2', '3'):
                        raise ValueError('仅使用 GPU 0、2、3；GPU 1 保留给原服务')
                    gpu_lock = (ROOT / f'experiments/functional-v1-20261104/gpu-{self.gpu}.lock').open('a')
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    memory = int(subprocess.check_output(['nvidia-smi', '-i', self.gpu,
                        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
                    if memory >= 512:
                        raise RuntimeError(f'GPU {self.gpu} 正被其他任务占用，已保留队列。')
                for folder in folders:
                    job = read(folder / 'job.json')
                    job.update(status='loading', message='正在载入四个候选的权重')
                    write(folder / 'job.json', job)
                with self.lock:
                    self.current = [f.name for f in folders]
                    self.running_since = time.monotonic()
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.gpu if self.device == 'cuda' else '',
                    HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONPATH=str(ROOT / 'scripts'),
                    OMP_NUM_THREADS='20', MKL_NUM_THREADS='20')
                log_path = RUN / f'worker-{uuid.uuid4().hex}.log'
                started_at = stamp()
                with log_path.open('w') as log:
                    self.process = subprocess.Popen([self.python, '-u', '-m', 'comparison_studio.worker',
                        '--device', self.device, '--max-seconds', str(max(1, self.cap - self.used - 5)),
                        *[str(f) for f in folders]], cwd=ROOT, env=env,
                        stdout=log, stderr=subprocess.STDOUT)
                    while self.process.poll() is None:
                        elapsed = time.monotonic() - self.running_since
                        if self.device == 'cuda':
                            write(self.ledger_path, dict(self.ledger, gpu_seconds=self.used + elapsed,
                                active_pid=self.process.pid, heartbeat_at=stamp()))
                            if elapsed >= max(0, self.cap - self.used - 5):
                                self.process.terminate()
                                try:
                                    self.process.wait(timeout=3)
                                except subprocess.TimeoutExpired:
                                    self.process.kill()
                                break
                        if self.stopped:
                            self.process.terminate()
                            break
                        time.sleep(.5)
                    self.process.wait(timeout=10)
                for folder in folders:
                    if read(folder / 'job.json')['status'] != 'completed':
                        self.fail(folder, '推理未完成或预算到达上限，详情已记录；不会用原图冒充结果。')
                with self.lock:
                    elapsed = time.monotonic() - self.running_since
                    if self.device == 'cuda':
                        self.used += elapsed
                    self.ledger['gpu_seconds'] = self.used
                    self.ledger['runs'].append(dict(jobs=self.current, device=self.device,
                        started_at=started_at, ended_at=stamp(), seconds=elapsed,
                        exit_code=self.process.returncode, log=log_path.name))
                    write(self.ledger_path, self.ledger)
            except (BlockingIOError, RuntimeError) as error:
                # Occupied cards are never taken from another process.
                for folder in folders:
                    job = read(folder / 'job.json')
                    if job['status'] in ('queued', 'loading'):
                        job.update(status='queued', message=str(error))
                        write(folder / 'job.json', job)
                self.wake.wait(5)
                self.wake.clear()
            except Exception as error:
                import traceback
                traceback.print_exc()
                for folder in folders:
                    self.fail(folder, f'启动失败：{type(error).__name__}: {error}')
            finally:
                if gpu_lock:
                    gpu_lock.close()
                with self.lock:
                    self.running_since = None
                    self.current = []
                self.process = None


@app.before_request
def same_origin():
    if request.method == 'POST' and request.headers.get('Origin'):
        if request.headers['Origin'].rstrip('/') != request.host_url.rstrip('/'):
            abort(403)


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/api/config')
def config():
    return jsonify(models=MODELS, samples=[dict(id=c['id'], title=c['title'], category=c['category'],
        instructions=c['instructions'], image=f"/samples/{c['id']}.png") for c in samples().values()],
        runtime=runner.health() if runner else {'device': 'unavailable'},
        protocol=dict(name='相同输入，原配套方案', languages=['zh', 'en'], training_use=False,
            note='R1保留首轮提示与编辑器；其余使用R15提示与编辑器。比较完整方案，不单独归因于权重。'))


@app.get('/health')
def health():
    return jsonify(status='ready', **(runner.health() if runner else {}))


@app.get('/samples/<sample_id>.png')
def sample_image(sample_id):
    item = samples().get(sample_id)
    if not item:
        abort(404)
    return send_file(ROOT / item['image'], mimetype='image/png')


def create_job(image, instructions, source, title, sample_id=None):
    folder = JOBS / uuid.uuid4().hex
    folder.mkdir(parents=True)
    image.save(folder / 'input.png', format='PNG', compress_level=1)
    job = dict(id=folder.name, title=title[:120], sample_id=sample_id, created_at=stamp(),
        status='queued', message='已加入对比队列', source=source, training_use=False,
        image_size=list(image.size), input_sha256=sha(folder / 'input.png'),
        instructions=instructions, results=initial_results(), models=list(MODELS),
        protocol='same-input-native-pipelines-v1')
    write(folder / 'job.json', job)
    return job


@app.post('/api/jobs')
def submit():
    instructions = {l: request.form.get(l, '').strip() for l in ('zh', 'en')}
    if any(not text or len(text) > 3000 for text in instructions.values()):
        return jsonify(error='请分别填写中文和英文指令，每项不超过3000字。'), 400
    if runner and runner.health().get('gpu_budget_exhausted'):
        return jsonify(error='本次 GPU 预算已用完，已有结果仍可查看。'), 409
    sample_id = request.form.get('sample_id')
    selected = samples().get(sample_id) if sample_id else None
    if selected:
        image_source = ROOT / selected['image']
        title = selected['title']
    elif 'image' in request.files:
        image_source = request.files['image'].stream
        title = Path(request.files['image'].filename or '我的照片').name
    else:
        return jsonify(error='请上传图片或选择一个示例。'), 400
    with submit_lock:
        pending = sum(read(p)['status'] in ('queued', 'loading', 'running') for p in JOBS.glob('*/job.json'))
        if pending >= 12:
            return jsonify(error='当前队列已满，请等待已有对比完成。'), 429
        with Image.open(image_source) as image:
            if image.width * image.height > 40_000_000:
                return jsonify(error='图片不能超过4000万像素。'), 400
            image = ImageOps.exif_transpose(image).convert('RGB')
            source = 'agent_smoke_test' if request.headers.get('X-IEA-Source') == 'agent-smoke' else (
                'historical_demo' if selected else 'user_upload')
            job = create_job(image, instructions, source, title, sample_id if selected else None)
    if runner:
        runner.wake.set()
    return jsonify(public(job)), 202


@app.get('/api/jobs')
def list_jobs():
    jobs = sorted([read(p) for p in JOBS.glob('*/job.json')], key=lambda j: j['created_at'], reverse=True)
    return jsonify(jobs=[dict(id=j['id'], title=j['title'], status=j['status'], created_at=j['created_at'],
                             source=j['source'], sample_id=j.get('sample_id')) for j in jobs[:100]])


@app.get('/api/jobs/<job_id>')
def get_job(job_id):
    return jsonify(public(read(job_path(job_id) / 'job.json')))


@app.post('/api/jobs/<job_id>/feedback')
def feedback(job_id):
    folder = job_path(job_id)
    value = request.get_json(silent=True)
    if not isinstance(value, dict):
        return jsonify(error='反馈格式无效'), 400
    ratings = value.get('ratings', {})
    preferred = value.get('preferred', {})
    if not isinstance(ratings, dict) or not isinstance(preferred, dict):
        return jsonify(error='反馈格式无效'), 400
    allowed = {f'{m}-{l}' for m in MODELS for l in ('zh', 'en')}
    if any(k not in allowed or v not in ('completed', 'partial', 'failed', 'uncertain', '') for k, v in ratings.items()):
        return jsonify(error='评分无效'), 400
    if any(k not in ('zh', 'en') or v not in (*MODELS, 'none', 'tie', '') for k, v in preferred.items()):
        return jsonify(error='偏好无效'), 400
    actor = 'agent_smoke_test' if request.headers.get('X-IEA-Source') == 'agent-smoke' else 'user'
    record = dict(ratings=ratings, preferred=preferred, comment=str(value.get('comment', ''))[:4000],
                  at=stamp(), source=actor, training_use=False, job_id=job_id)
    with feedback_lock:
        write(folder / 'feedback.json', record)
        append(RUN / 'feedback-history.jsonl', record)
    return jsonify(status='saved', feedback=record)


@app.get('/files/<job_id>/<filename>')
def files(job_id, filename):
    folder = job_path(job_id)
    allowed = {'input.png', *[f'{m}-{l}.png' for m in MODELS for l in ('zh', 'en')]}
    if filename not in allowed:
        abort(404)
    return send_from_directory(folder, filename, as_attachment=request.args.get('download') == '1')


@app.get('/api/jobs/<job_id>/download')
def download(job_id):
    folder = job_path(job_id)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in folder.iterdir():
            if path.suffix in ('.json', '.png'):
                z.write(path, path.name)
    stream.seek(0)
    return send_file(stream, mimetype='application/zip', as_attachment=True, download_name=f'IEA-compare-{job_id[:8]}.zip')


@app.errorhandler(ValueError)
@app.errorhandler(UnidentifiedImageError)
@app.errorhandler(Image.DecompressionBombError)
def bad_input(error):
    return jsonify(error=f'输入无法处理：{error}'), 400


@app.errorhandler(FileNotFoundError)
def missing(_):
    return jsonify(error='记录不存在'), 404


@app.errorhandler(413)
def large(_):
    return jsonify(error='上传文件超过22 MB。'), 413


def main():
    global runner
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=12348)
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--gpu-budget-hours', type=float, default=0)
    ap.add_argument('--worker-python', default=str(ROOT / 'envs/iea-rl/bin/python'))
    args = ap.parse_args()
    if args.device == 'cuda' and args.gpu_budget_hours <= 0:
        ap.error('GPU mode requires an explicitly authorized positive budget')
    JOBS.mkdir(parents=True, exist_ok=True)
    runner = Runner(args.worker_python, args.device, args.gpu, args.gpu_budget_hours)
    atexit.register(runner.close)
    app.run(host='127.0.0.1', port=args.port, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()

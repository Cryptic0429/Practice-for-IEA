"""Loopback-only model worker, using the training environment and exact runtime."""
from __future__ import annotations
import argparse
import gc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sys
import threading
import time
import traceback
import select
import subprocess
from .common import ROOT, EXP, DATA, GPUJob, write, sha, stamp, EDITOR_VERSION
from .catalog import models, resolve


class Engine:
    def __init__(self, gpu):
        self.gpu = gpu
        self.lock = threading.Lock()
        self.process = self.model_id = None
        self.last_used = time.monotonic()
        threading.Thread(target=self.reaper, daemon=True).start()

    def unload(self, error=None):
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write(json.dumps({'command': 'quit'})+'\n')
                    self.process.stdin.flush()
                    self.process.wait(timeout=30)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    self.process.terminate()
                    self.process.wait(timeout=30)
            self.process.stdin.close()
            self.process.stdout.close()
        self.process = self.model_id = None

    def receive(self, timeout=180):
        if not select.select([self.process.stdout], [], [], timeout)[0]:
            raise TimeoutError('Timed out waiting for the owned GPU worker')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('GPU worker exited; inspect its local log')
        return json.loads(line)

    def reaper(self):
        while True:
            time.sleep(10)
            with self.lock:
                if self.process is not None and time.monotonic()-self.last_used > 120:
                    self.unload()

    def infer(self, value):
        if (EXP/'BUDGET_STOP.json').exists():
            raise RuntimeError('首轮实验的GPU预算已用完，当前停止新推理，已保存结果仍可查看。')
        image = Path(value['image']).resolve()
        upload_root = (EXP/'demo/requests').resolve()
        if not image.is_relative_to(upload_root) or image.name != 'input.png':
            raise ValueError('Image path is outside the trial upload directory')
        instruction = value['instruction'].strip()
        if not instruction or len(instruction) > 6000:
            raise ValueError('A nonempty editing instruction is required (maximum 6000 characters)')
        spec = resolve(value['model_id'])
        with self.lock:
            if self.process is not None and self.process.poll() is not None:
                self.unload()
            if self.model_id != spec['id']:
                self.unload()
                used = int(subprocess.check_output(['nvidia-smi','-i',self.gpu,
                    '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                if used >= 512:
                    raise RuntimeError('GPU正在运行本轮实验，当前任务结束后可以继续试用。')
                self.process = subprocess.Popen([sys.executable, '-u', '-m', 'functional_v1.gpu_child',
                    '--gpu', self.gpu, '--model', spec['id']], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    text=True, bufsize=1)
                try:
                    ready = self.receive()
                    if ready.get('status') != 'ready':
                        raise RuntimeError('GPU child did not become ready')
                    self.model_id = spec['id']
                except BaseException as error:
                    self.unload(error)
                    raise
            try:
                self.process.stdin.write(json.dumps({'image': str(image), 'instruction': instruction})+'\n')
                self.process.stdin.flush()
                return self.receive()
            finally:
                self.last_used = time.monotonic()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', choices=['0','2','3'], default='0')
    ap.add_argument('--port', type=int, default=18282)
    args = ap.parse_args()
    engine = Engine(args.gpu)
    class Handler(BaseHTTPRequestHandler):
        def reply(self, value, status=200):
            content = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path == '/health':
                self.reply({'status': 'ready', 'loaded_model': engine.model_id,
                            'gpu': args.gpu, 'idle_unload_seconds': 120})
            elif self.path == '/models':
                self.reply({'models': models()})
            else:
                self.reply({'error': 'not_found'}, 404)

        def do_POST(self):
            if self.path == '/unload':
                with engine.lock:
                    engine.unload()
                return self.reply({'status': 'unloaded'})
            if self.path != '/infer':
                return self.reply({'error': 'not_found'}, 404)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length < 65536:
                    raise ValueError('Invalid request size')
                value = json.loads(self.rfile.read(length))
                self.reply(engine.infer(value))
            except Exception as error:
                traceback.print_exc()
                self.reply({'status': 'failed', 'error': f'{type(error).__name__}: {error}'}, 422)
    print(f'Functional inference worker ready at 127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()

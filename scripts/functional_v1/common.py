from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

ROOT = Path(os.environ.get('IEA_ROOT', Path(__file__).resolve().parents[2]))
NAME = 'functional-v1-20261104'
DATA = ROOT / 'data/training' / NAME
EXP = ROOT / 'experiments' / NAME
BASE = ROOT / 'models/Qwen2.5-VL-7B-Instruct'
PARENT = ROOT / 'models/from-sft-standard/sft-calibrated/seed-02'
TOOLS = ('exposure', 'brightness', 'contrast', 'highlights', 'shadows',
         'saturation', 'temperature', 'tint', 'sepia')
EDITOR_VERSION = 'functional-editor-v1.2-sepia'
EVAL = EXP / 'evaluation' / EDITOR_VERSION


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_parameter:' + key)
        result[key] = value
    return result


def validate(parameters):
    if not isinstance(parameters, dict):
        raise ValueError('parameters_must_be_an_object')
    for key, value in parameters.items():
        if key not in TOOLS:
            raise ValueError('unsupported_parameter:' + key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('parameter_must_be_numeric:' + key)
        if not math.isfinite(value) or not -100 <= value <= 100:
            raise ValueError('parameter_out_of_range:' + key)
        if key == 'sepia' and value < 0:
            raise ValueError('sepia_strength_must_be_between_0_and_100')
    return dict(parameters)


def parse(response):
    match = re.fullmatch(r'\s*```json\s*([\s\S]*?)\s*```\s*', response)
    if not match:
        raise ValueError('expected_one_json_code_block_without_other_text')
    return validate(json.loads(match.group(1), object_pairs_hook=_pairs))


def image_path(case):
    return ROOT / case['image']


def system_prompt():
    return (
        'You edit the supplied photograph according to the user request. '
        'Follow the requested change, including its direction and strength. '
        'Preserve unrelated content and avoid clipped highlights, crushed shadows, and unintended colour casts. '
        'Return exactly one fenced json code block with a flat object of numerical editing parameters. '
        'Use only: ' + ', '.join(TOOLS) + '. All values range from -100 to 100; omitted or zero means no change. '
        'Positive exposure/brightness lighten; negative darken. Positive contrast increases contrast, '
        'negative decreases it. Positive highlights/shadows lift the respective tonal range; '
        'negative lowers it. Positive saturation increases colour intensity. '
        'Positive temperature warms (more red, less blue); negative cools. '
        'Positive tint adds green; negative tint adds magenta. '
        'The additional sepia tool makes an overall brown/sepia vintage rendering; '
        'its strength ranges from 0 (none) to 100 (fully sepia). '
        'Warm white balance alone does not turn green foliage brown. '
        'Do not add generic enhancement that was not requested. '
        'If explicitly asked to keep the image unchanged, return {}. '
        'Chinese and English requests have equal priority. Do not output explanations.'
    )


def input_row(case, language):
    """Allowlist: references, labels and review notes cannot enter the model."""
    path = str(image_path(case))
    return {'images': [path], 'extra_info': {'origin_image_path': path},
            'conversations': [{'role': 'system', 'content': system_prompt()},
                              {'role': 'user', 'content': '<image>\n' + case['instructions'][language]}]}


class BudgetExceeded(RuntimeError):
    pass


class GPUJob:
    """Cross-process occupied-GPU wall-time ledger, including loading and waiting."""
    def __init__(self, gpu, stage, max_seconds=7200):
        self.gpu, self.stage, self.cap = str(gpu), stage, max_seconds
        if self.gpu not in ('0', '2', '3'):
            raise ValueError('GPU 1 is reserved for the existing IEA service')
        self.id = uuid.uuid4().hex

    @contextlib.contextmanager
    def ledger(self):
        EXP.mkdir(parents=True, exist_ok=True)
        with (EXP / 'resource.lock').open('a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            path = EXP / 'resources.json'
            value = read(path) if path.exists() else {'limit_gpu_hours': 16, 'jobs': []}
            yield value
            write(path, value)

    @staticmethod
    def consumed(value):
        now = time.time()
        return sum((j.get('ended_epoch', now) - j['started_epoch']) / 3600 for j in value['jobs'])

    def __enter__(self):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != self.gpu:
            raise ValueError('Set CUDA_VISIBLE_DEVICES explicitly to the assigned physical GPU')
        self.lock = (EXP / f'gpu-{self.gpu}.lock')
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.lock.open('a')
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            used = int(subprocess.check_output(['nvidia-smi', '-i', self.gpu,
                       '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
            if used >= 512:
                raise RuntimeError(f'GPU {self.gpu} is occupied ({used} MiB); do not take it over')
            self.started = time.time()
            with self.ledger() as value:
                if self.consumed(value) >= value['limit_gpu_hours']:
                    raise BudgetExceeded('16 GPU hour experiment budget exhausted')
                value['jobs'].append({'id': self.id, 'pid': os.getpid(), 'gpu': self.gpu,
                                      'stage': self.stage, 'editor': EDITOR_VERSION, 'started_at': stamp(),
                                      'started_epoch': self.started, 'heartbeat_epoch': self.started,
                                      'max_seconds': self.cap, 'status': 'running'})
        except BaseException:
            self.handle.close()
            raise
        return self

    def tick(self):
        with self.ledger() as value:
            next(j for j in value['jobs'] if j['id'] == self.id)['heartbeat_epoch'] = time.time()
            exhausted = self.consumed(value) >= value['limit_gpu_hours']
        if exhausted or time.time() - self.started >= self.cap:
            raise BudgetExceeded('Experiment or job GPU-time limit reached; preserve partial results')

    def __exit__(self, kind, error, traceback):
        with self.ledger() as value:
            job = next(j for j in value['jobs'] if j['id'] == self.id)
            job.update(ended_epoch=time.time(), ended_at=stamp(),
                       status='completed' if kind is None else 'stopped', error=str(error) if error else None)
            value['consumed_gpu_hours'] = self.consumed(value)
        self.handle.close()


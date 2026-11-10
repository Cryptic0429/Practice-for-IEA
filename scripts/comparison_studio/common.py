from pathlib import Path
from functional_v1.common import ROOT, read, write, sha, stamp, append

RUN = ROOT / 'experiments/comparison-studio-20261110'
JOBS = RUN / 'jobs'
MODELS = {
    'r1': dict(name='R1', subtitle='首轮默认 · seed 20261104',
        adapter='models/from-sft-calibrated/sft-functional/seed-01',
        sha256='54161bd9c9f2e3aa518572be6cf1dad81f8bebe33d92d31107d64d1f3c4435b0',
        editor='functional-editor-v1.2-sepia', note='冻结验收 31/50；力度与复合要求仍有限'),
    'r15': dict(name='R15', subtitle='第1轮 · seed 20261127',
        adapter='models/from-functional-v11/sft-shadow-correction/seed-01',
        sha256='97c84f1799fb319799c86515402574b3b585593763e5388eb30d0df3e0d6ea90',
        editor='functional-editor-v15-shared-shadow-gain-candidate', note='原始开发 16/20；受控 75/100，有严重偏色'),
    'r23': dict(name='R23', subtitle='直接 LoRA · seed 20261208',
        adapter='models/from-functional-v15/sft-quality-coverage/seed-20261208',
        sha256='043489c53d6cc50625887665bf18edbe0f62c9aeb0fb16c0f481f3ead37af838',
        editor='functional-editor-v15-shared-shadow-gain-candidate', note='原始开发 16/20；历史回归 4/6，最终未验收'),
    'r21': dict(name='R15 + R21', subtitle='幅度与质量控制 · seed 20261205',
        adapter='models/from-functional-v11/sft-shadow-correction/seed-01',
        sha256='97c84f1799fb319799c86515402574b3b585593763e5388eb30d0df3e0d6ea90',
        controller='models/from-functional-v15/semantic-preserving-controller/seed-01/controller.pt',
        controller_sha256='838342a2167038589a91b646ad98432ec4749e698d4b29ad7f2b520b366a26da',
        editor='functional-editor-v15-shared-shadow-gain-candidate', note='受控开发 82/100；换种子复核失败'),
}
SAMPLE_NAMES = {
    'native-dev-001': ('棕褐色人像', '色调'),
    'native-dev-006': ('室内偏色', '白平衡'),
    'native-dev-011': ('逆光与肤色', '复合要求'),
    'native-dev-023': ('人像层次', '对比度'),
    'native-dev-070': ('星空与噪声', '夜景提亮'),
    'native-dev-161': ('山湖与倒影', '整体压暗'),
}

def samples():
    path = ROOT / 'data/training/functional-v6-20261105/development-native-cases.json'
    return {c['id']: dict(c, title=SAMPLE_NAMES[c['id']][0], category=SAMPLE_NAMES[c['id']][1])
            for c in read(path) if c['id'] in SAMPLE_NAMES}

def initial_results():
    return [dict(model=m, language=l, status='queued', after=None) for m in MODELS for l in ('zh', 'en')]

def job_path(job_id):
    import re
    if not isinstance(job_id, str) or not re.fullmatch(r'[0-9a-f]{32}', job_id):
        raise ValueError('无效的记录编号')
    p = JOBS / job_id
    if not (p / 'job.json').is_file():
        raise FileNotFoundError(job_id)
    return p

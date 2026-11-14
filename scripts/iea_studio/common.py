from pathlib import Path
import re
from functional_v1.common import ROOT, read, write, sha, stamp, append

RUN = ROOT / 'experiments/comparison-studio-20261110'
JOBS = RUN / 'jobs'
SESSIONS = RUN / 'sessions'
CORE_MODEL = 'r21'
BASELINES = ('r15', 'r1')
MODELS = {
    'r21': dict(name='IEA 智能调色核心', tier='core', badge='核心引擎',
        subtitle='语义决策主干与幅度风险控制头协同架构',
        adapter='models/from-functional-v11/sft-shadow-correction/seed-01',
        sha256='97c84f1799fb319799c86515402574b3b585593763e5388eb30d0df3e0d6ea90',
        controller='models/from-functional-v15/semantic-preserving-controller/seed-01/controller.pt',
        controller_sha256='838342a2167038589a91b646ad98432ec4749e698d4b29ad7f2b520b366a26da',
        editor='functional-editor-v15-shared-shadow-gain-candidate',
        note='默认主力方案：先锁定语义工具与方向，再融合图像纹理特征校准调色幅度与画质风险'),
    'r15': dict(name='无控制头消融版', tier='baseline', badge='消融对照',
        subtitle='仅启用语义决策主干与阴影增益编辑器',
        adapter='models/from-functional-v11/sft-shadow-correction/seed-01',
        sha256='97c84f1799fb319799c86515402574b3b585593763e5388eb30d0df3e0d6ea90',
        editor='functional-editor-v15-shared-shadow-gain-candidate',
        note='用于对照验证控制头在幅度校准与过曝、噪点风险拦截上的真实增益'),
    'r1': dict(name='早期功能基线', tier='baseline', badge='历史基线',
        subtitle='初代双语指令参数解析与基础编辑器',
        adapter='models/from-sft-calibrated/sft-functional/seed-01',
        sha256='54161bd9c9f2e3aa518572be6cf1dad81f8bebe33d92d31107d64d1f3c4435b0',
        editor='functional-editor-v1.2-sepia',
        note='保留初代提示协议与编辑器作为历史对照基准'),
}
SAMPLE_NAMES = {
    'native-dev-001': ('棕褐色人像', '色调风格'),
    'native-dev-006': ('室内偏色校正', '白平衡'),
    'native-dev-011': ('逆光与肤色保留', '复合光影'),
    'native-dev-023': ('人像层次增强', '对比度'),
    'native-dev-070': ('星空与夜景提亮', '暗部风险'),
    'native-dev-161': ('山湖与倒影压暗', '曝光控制'),
}


def samples():
    path = ROOT / 'data/training/functional-v6-20261105/development-native-cases.json'
    return {c['id']: dict(c, title=SAMPLE_NAMES[c['id']][0], category=SAMPLE_NAMES[c['id']][1])
            for c in read(path) if c['id'] in SAMPLE_NAMES}


def initial_results():
    return [dict(model=m, language=l, status='queued', after=None) for m in MODELS for l in ('zh', 'en')]


def job_path(job_id):
    if not isinstance(job_id, str) or not re.fullmatch(r'[0-9a-f]{32}', job_id):
        raise ValueError('无效的记录编号')
    p = JOBS / job_id
    if not (p / 'job.json').is_file():
        raise FileNotFoundError(job_id)
    return p


def session_path(session_id):
    if not isinstance(session_id, str) or not re.fullmatch(r'[0-9a-f]{32}', session_id):
        raise ValueError('无效的会话编号')
    p = SESSIONS / session_id
    if not (p / 'session.json').is_file():
        raise FileNotFoundError(session_id)
    return p

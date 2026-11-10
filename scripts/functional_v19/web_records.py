"""Read-only round19 training and reviewed developer records; no acceptance UI."""
import json
import math
import re
from pathlib import Path

from flask import Blueprint, abort, jsonify, render_template_string, request, send_from_directory
from functional_v1.common import ROOT, read, GPUJob
from .outcomes import EDITOR_VERSION

EXP = ROOT / 'experiments/functional-v19-20261107'
DATA = ROOT / 'data/training/functional-v19-20261107'
EVAL = EXP / 'evaluation' / EDITOR_VERSION
bp = Blueprint('functional_v19_records', __name__)
SPLITS = ('native-development', 'old-native', 'development')



def overview():
    ledger = read(ROOT / 'experiments/functional-v1-20261104/resources.json')
    admission=read(DATA/'admission.json')
    shown=dict(admission,unique_semantic_inputs=admission['semantic_cases'],unique_language_inputs=admission['language_records'],original_groups=admission['source_groups'],source_domains=admission['domains'],edit_keep_decline=admission['actions'],unchanged_original_English_requests=admission['original_English'],AI_requests=admission['AI_requests'],anchor_rows_per_epoch=1850,total_presentations_per_epoch=2002,input_features_sha256=admission['input_features_sha256'])
    return dict(admission=shown, occupied_gpu_hours=GPUJob.consumed(ledger), budget_limit_gpu_hours=ledger["limit_gpu_hours"],
                training={p.parent.name: read(p) for p in (EXP / 'runs').glob('*/status.json')},
                summaries={split: {p.parent.name: read(p) for p in (EVAL / split).glob('*/direct-functional-summary.json') if read(p).get('status')=='completed'} for split in SPLITS},
                selection=read(EXP / 'development-selection.json') if (EXP / 'development-selection.json').exists() else None,
                decision_summaries={p.parent.name: read(p) for p in (EVAL / 'decision-development').glob('*/decision-summary.json')},
                review_mode='AI辅助直接视觉评审；用户实际可用性另记', acceptance_exposed=False)


@bp.get('/api/v19/experiment')
def experiment():
    return jsonify(overview())


def label_evidence(e):
    if e.get('selection',{}).get('native_observation'): return e['selection']['native_observation']['evidence']
    if e.get('selected_repair_native'): return e['selected_repair_native']['evidence']
    if e.get('native'): return e['native'].get('evidence','继承此前标签的原尺寸核验证据；不是本轮新模型结果。')
    if e.get('native_pixel_supplement'):return e['native_pixel_supplement']['evidence']
    if e.get('raw'):return e['raw']['evidence']
    if e.get('selected_repair'):return e['selected_repair']['evidence']
    if e.get('observation'):return e['observation']['evidence']
    if e.get('conclusion'):return e['conclusion'] + ' 已执行并查看的新尝试未找到合格结果；这不是穷尽参数空间的证明。'
    return '原样继承此前 AI 核验；原图、请求和目标像素一致，未解决的争议仍保留。这不是本轮重新独立评审。'


def records(kind, model=None, split='native-development'):
    rows = []
    if kind == 'training':
        for c in read(DATA / 'cases.json'):
            rows.append(dict(id=c['id'], domain=c['domain'], action=c['action'], instructions=c['instructions'],
                origin=c['instruction_origin'], status='实际采用的训练标签 · AI核验；不是模型预测',
                evidence='本轮原样继承此前核验的标签。应按以上要求改变照片，同时保留主体、构图、自然颜色与细节；这张图不代表本轮模型已经学会。',raw_evidence=label_evidence(c['label_evidence']),
                parameters=c.get('parameters'), before=f'/v19/training-image/{c["id"]}/before',
                after=f'/v19/training-image/{c["id"]}/after' if c.get('target') else None,
                no_result='训练决策：明确提示无法可靠完成，不返回结果图。' if c['action'] == 'decline' else None))
    elif kind == 'evaluation':
        if split not in SPLITS or not model or not re.fullmatch(r'[a-z0-9-]+', model):
            return rows
        folder = EVAL / split / model
        if not (folder / 'direct-review.json').exists() or not (folder/'direct-functional-summary.json').exists() or read(folder/'direct-functional-summary.json')['status']!='completed':
            return rows
        notes = {(r['case_id'], r['language']): r['assessment'] for r in read(folder / 'direct-review.json')}
        for r in map(json.loads, (folder / 'predictions.jsonl').read_text().splitlines()):
            n = notes.get((r['case_id'], r['language']))
            if n is None:
                continue
            prefix = f'/v19/evaluation-image/{split}/{model}/{r["case_id"]}/{r["language"]}'
            completion = dict(completed='动作完成', partial='部分完成', failed='失败', uncertain='不确定')
            quality = dict(acceptable='画质合格', minor_issue='轻微副作用', major_issue='重大副作用', uncertain='画质不确定')
            judgment = ('pass' if n['completion']=='completed' and n['quality']=='acceptable' else
                        'fail' if n['completion'] in ('partial','failed') or n['quality'] in ('minor_issue','major_issue') else 'uncertain')
            rows.append(dict(id=r['case_id'] + '-' + r['language'], domain=r['domain'], action=r['status'],
                instructions={'zh': r['instruction'], 'en': ''}, origin=model + ' · ' + split + (' · 既有冻结对照预测，原样复用，无新增推理' if model == 'parent-editor-only' else ''),
                status=completion[n['completion']] + ' / ' + quality[n['quality']], evidence=n['evidence'],
                before=prefix + '/before', after=prefix + '/after' if r['status'] == 'executed' else None,
                judgment=judgment, parameters=r.get('parameters'),
                no_result='未返回编辑图；原修图要求计为未完成。' if r['status'] != 'executed' else None))
    return rows


@bp.get('/v19')
def page():
    kind = request.args.get('kind', 'training')
    if kind not in ('training', 'evaluation'):
        abort(404)
    domain, action = request.args.get('domain', 'all'), request.args.get('action', 'all')
    if kind!='training':action='all'
    judgment = request.args.get('judgment', 'all')
    split = request.args.get('split', 'native-development')
    if split not in SPLITS:abort(404)
    summaries = overview()
    model_ids = sorted(summaries['summaries'].get(split, {}))
    model = request.args.get('model') or (model_ids[0] if model_ids else '')
    rows = [r for r in records(kind, model, split) if (domain == 'all' or r['domain'] == domain) and (action == 'all' or r['action'] == action)
            and (kind!='evaluation' or judgment=='all' or r['judgment']==judgment)]
    page_number = request.args.get('page', 1, type=int)
    pages = max(1, math.ceil(len(rows) / 12))
    page_number = min(max(1, page_number), pages)
    query = dict(kind=kind, domain=domain, action=action, split=split, model=model,judgment=judgment)
    from urllib.parse import urlencode
    return render_template_string(Path(__file__).with_name('records.html').read_text(),
        **summaries, rows=rows[(page_number - 1) * 12:page_number * 12], total=len(rows),
        kind=kind, domain=domain, action=action, model=model, model_ids=model_ids, split=split,judgment=judgment,
        page_number=page_number, pages=pages,
        previous_url='/v19?' + urlencode(dict(query, page=page_number - 1)),
        next_url='/v19?' + urlencode(dict(query, page=page_number + 1)))


@bp.get('/v19/training-image/<cid>/<role>')
def training_image(cid, role):
    c = next((c for c in read(DATA / 'cases.json') if c['id'] == cid), None)
    if not c or role not in ('before', 'after') or role == 'after' and not c.get('target'):
        abort(404)
    path = ROOT / c['image' if role == 'before' else 'target']
    return send_from_directory(path.parent, path.name)


@bp.get('/v19/evaluation-image/<split>/<model>/<cid>/<language>/<role>')
def evaluation_image(split, model, cid, language, role):
    if split not in SPLITS or not re.fullmatch(r'[a-z0-9-]+', model) or language not in ('zh', 'en') or role not in ('before', 'after'):
        abort(404)
    folder = EVAL / split / model
    if not (folder / 'direct-review.json').exists() or not (folder/'direct-functional-summary.json').exists() or read(folder/'direct-functional-summary.json')['status']!='completed':
        abort(404)
    r = next((r for r in map(json.loads, (folder / 'predictions.jsonl').read_text().splitlines()) if r['case_id'] == cid and r['language'] == language), None)
    if not r or role == 'after' and r['status'] != 'executed':
        abort(404)
    path = ROOT / r['input_image' if role == 'before' else 'output_image']
    return send_from_directory(path.parent, path.name)

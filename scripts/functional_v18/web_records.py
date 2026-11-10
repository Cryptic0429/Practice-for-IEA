"""Read-only round18 training and reviewed developer records; no acceptance UI."""
import json
import math
import re
from pathlib import Path

from flask import Blueprint, abort, jsonify, render_template_string, request, send_from_directory
from functional_v1.common import ROOT, read, GPUJob
from .outcomes import EDITOR_VERSION

EXP = ROOT / 'experiments/functional-v18-20261107'
DATA = ROOT / 'data/training/functional-v18-20261107'
EVAL = EXP / 'evaluation' / EDITOR_VERSION
bp = Blueprint('functional_v18_records', __name__)
SPLITS = ('native-development', 'old-native', 'development')
PUBLIC_REASONS = {
    'train-020-restore-exposure-mild': '适度提亮后，人物、蓝色衣服和摩托车更清楚，暗部纹理仍保留。过强结果使肤色、草地和浅色背景发白，显得不自然。',
    'train-020-restore-exposure-strong': '这张输入明显偏暗，需要更充分地提亮。采用结果保留了衣纹、发动机和车轮结构；较弱结果仍让人物和摩托车偏暗。',
    'train-066-restore-exposure-mild': '采用结果适度提亮面部，保留雀斑、眼睛和发丝。过强结果让额头与脸颊明显偏黄橙，衣服也过亮。',
    'train-066-restore-exposure-strong': '采用结果恢复面部亮度，同时保留五官、雀斑和头发层次。较弱结果仍让面部和头发偏暗。',
    'train-210-restore-exposure-mild': '采用结果提亮了山谷、河流和前景，同时保留浅色云层的渐变。过强结果让天空更白、山景颜色和层次变平。',
    'train-210-restore-exposure-strong': '采用结果让前景植物、河流和山坡重新清楚可见，并保留云层与积雪结构。较弱结果仍显得灰暗。',
    'train-021-restore-exposure-strong': '采用结果恢复水面、云层与岩石的可见层次，波纹和岩石结构仍在。较弱结果让岛石正面和右侧山坡继续偏暗。',
    'train-048-restore-contrast-mild': '采用结果恢复树皮、裤褶和衣服文字的明暗分离。过强结果压黑头发与树影，使脸颊偏橙、浅色地面刺眼。',
    'train-048-restore-contrast-strong': '采用结果去掉明显灰雾，同时保留衣纹和树皮纹理。较弱结果仍灰蒙蒙，脸部和裤子的层次不足。',
    'train-042-restore-contrast-mild': '采用结果保留白衬衣褶皱、砖墙与草地纹理。过强结果使阳光下的草地过黄过亮，白衬衣层次变平。',
    'train-042-restore-contrast-strong': '采用结果恢复砖缝、草地和衣服的明暗层次。较弱结果仍有灰雾，墙面和人物显得平淡。',
    'train-100-restore-contrast-mild': '采用结果自然地区分花朵、棕榈叶和纪念碑的明暗。过强结果让暗部发黑、天空过青，花朵高光显得生硬。',
    'train-100-restore-contrast-strong': '采用结果清除灰雾，保留花瓣和叶片结构。较弱结果仍缺乏明暗分离，整幅画面偏灰。',
    'train-120-restore-contrast-mild': '采用结果让路面石粒更有层次，并保留天空和树林暗部。过强结果让树丛内部压黑、天空近乎白板。',
    'train-120-restore-contrast-strong': '采用结果恢复路面石粒、白线和树林的明暗层次。较弱结果仍有灰雾；这里没有宣称修复原照片的失焦。',
    'train-110-restore-saturation-mild': '采用结果恢复气球和植被的颜色，保留儿童肤色与白裙褶皱。过强结果使青、粉、绿过艳，裙摆出现不自然的紫色。',
    'train-110-restore-saturation-strong': '采用结果恢复气球和植被的颜色，并保留裙子和面部结构。较弱结果仍明显褪色。',
    'train-064-restore-saturation-mild': '采用结果适度恢复蓝色衣服、栗色马匹和草地颜色。过强结果让蓝色过艳，肤色和马匹偏橙。',
    'train-064-restore-saturation-strong': '采用结果恢复蓝色衣服、栗色马匹和植被，保留毛发与围栏结构。较弱结果仍灰淡；原有人脸偏亮的问题没有被宣称局部修复。',
}


def overview():
    ledger = read(ROOT / 'experiments/functional-v1-20261104/resources.json')
    return dict(admission=read(DATA / 'admission.json'), occupied_gpu_hours=GPUJob.consumed(ledger), budget_limit_gpu_hours=ledger["limit_gpu_hours"],
                training={p.parent.name: read(p) for p in (EXP / 'runs').glob('*/status.json')},
                summaries={split: {p.parent.name: read(p) for p in (EVAL / split).glob('*/direct-functional-summary.json') if read(p).get('status')=='completed'} for split in SPLITS},
                selection=read(EXP / 'development-selection.json') if (EXP / 'development-selection.json').exists() else None,
                decision_summaries={p.parent.name: read(p) for p in (EVAL / 'decision-development').glob('*/decision-summary.json')},
                review_mode='AI辅助直接视觉评审；用户实际可用性另记', acceptance_exposed=False)


@bp.get('/api/v18/experiment')
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
                evidence=label_evidence(c['label_evidence']),
                parameters=c.get('parameters'), before=f'/v18/training-image/{c["id"]}/before',
                after=f'/v18/training-image/{c["id"]}/after' if c.get('target') else None,
                no_result='训练决策：明确提示无法可靠完成，不返回结果图。' if c['action'] == 'decline' else None))
    elif kind == 'preferences':
        cases={c['id']:c for c in read(DATA/'cases.json')}
        seen=set()
        for item in read(DATA/'preferences.json'):
            cid=item['case_id']
            if cid in seen:continue
            seen.add(cid);c=cases[cid]
            rows.append(dict(id=cid,domain=c['domain'],action='preference',instructions=c['instructions'],
                origin='AI构造的全局修复要求与明确标记的合成退化；不是实际试用反馈。',status='本轮采用的图像偏好 · 已按实际图片核验；不是新模型预测',
                evidence=PUBLIC_REASONS[cid],raw_evidence=item['native_observation']['evidence'],parameters=item['chosen']['parameters'],
                rejected_parameters=item['rejected']['parameters'],before=f'/v18/preference-image/{cid}/before',
                after=f'/v18/preference-image/{cid}/after',rejected=f'/v18/preference-image/{cid}/rejected'))
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
            prefix = f'/v18/evaluation-image/{split}/{model}/{r["case_id"]}/{r["language"]}'
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


@bp.get('/v18')
def page():
    kind = request.args.get('kind', 'training')
    if kind not in ('training', 'preferences', 'evaluation'):
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
        previous_url='/v18?' + urlencode(dict(query, page=page_number - 1)),
        next_url='/v18?' + urlencode(dict(query, page=page_number + 1)))


@bp.get('/v18/training-image/<cid>/<role>')
def training_image(cid, role):
    c = next((c for c in read(DATA / 'cases.json') if c['id'] == cid), None)
    if not c or role not in ('before', 'after') or role == 'after' and not c.get('target'):
        abort(404)
    path = ROOT / c['image' if role == 'before' else 'target']
    return send_from_directory(path.parent, path.name)


@bp.get('/v18/preference-image/<cid>/<role>')
def preference_image(cid,role):
    if role not in ('before','after','rejected'):abort(404)
    item=next((p for p in read(DATA/'preferences.json') if p['case_id']==cid),None)
    if item is None:abort(404)
    if role=='before':
        case=next(c for c in read(DATA/'cases.json') if c['id']==cid)
        path=ROOT/case['image']
    else:path=ROOT/item['chosen_image' if role=='after' else 'rejected_image']
    return send_from_directory(path.parent,path.name)


@bp.get('/v18/evaluation-image/<split>/<model>/<cid>/<language>/<role>')
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

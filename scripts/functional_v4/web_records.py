"""Actual round-four training targets and reviewed development outputs; no acceptance access."""
import json
from flask import Blueprint, jsonify, request, send_from_directory
from .common import ROOT, DATA, EXP, EVAL, read

bp = Blueprint('functional_v4_records', __name__)
MODELS = ('parent-plain', 'parent-measured', 'plain-e1', 'plain-e2', 'measured-e1', 'measured-e2')


@bp.get('/api/v4/experiment')
def experiment():
    s = read(EXP/'development-selection.json')
    return jsonify(selection=s, admission=read(DATA/'admission.json'),
                   development_results=s['development_results'], natural_results=s['natural_results'],
                   resources=read(ROOT/'experiments/functional-v1-20261104/resources.json'))


@bp.get('/api/v4/training-examples')
def training():
    return jsonify(examples=[dict(id=c['id'], family=c['family'], domain=c['domain'], instructions=c['instructions'],
                     original_record_id=c['source_id'], parameters=c['parameters'],
                     instruction_origin='保留原始英文／AI中文翻译' if c['id'].startswith('natural-') else 'AI构造可控任务',
                     status='实际采用标签 · AI直接视觉核验',
                     before=f'/v4/training-image/{c["id"]}/before', after=f'/v4/training-image/{c["id"]}/after')
                   for c in read(DATA/'cases.json')])


@bp.get('/v4/training-image/<cid>/<role>')
def training_image(cid, role):
    if role not in ('before', 'after'):
        return jsonify(error='not_found'), 404
    c = next((c for c in read(DATA/'cases.json') if c['id'] == cid), None)
    if not c:
        return jsonify(error='not_found'), 404
    p = ROOT/c['image' if role == 'before' else 'target']
    return send_from_directory(p.parent, p.name)


@bp.get('/api/v4/evaluation-examples')
def examples():
    mid, split = request.args.get('model', 'measured-e1'), request.args.get('split', 'development')
    if mid not in MODELS or split not in ('development', 'natural-use'):
        return jsonify(error='not_found'), 404
    folder = EVAL/split/mid
    reviews = {(r['case_id'], r['language']): r['assessment'] for r in read(folder/'direct-review.json')}
    result = []
    for r in map(json.loads, (folder/'predictions.jsonl').read_text().splitlines()):
        prefix = f'/v4/evaluation-image/{split}/{mid}/{r["case_id"]}/{r["language"]}'
        result.append(dict(case_id=r['case_id'], language=r['language'], instruction=r['instruction'], model_id=mid,
                           split=split, assessment=reviews[r['case_id'], r['language']], before=prefix+'/before',
                           after=prefix+'/after' if r['status'] == 'executed' else None))
    rank = {'major_issue': 0, 'minor_issue': 1, 'uncertain': 2, 'acceptable': 3}
    result.sort(key=lambda r: (rank[r['assessment']['quality']], r['assessment']['completion'] == 'completed', r['case_id'], r['language']))
    return jsonify(examples=result)


@bp.get('/v4/evaluation-image/<split>/<mid>/<cid>/<language>/<role>')
def evaluation_image(split, mid, cid, language, role):
    if mid not in MODELS or split not in ('development', 'natural-use') or language not in ('zh', 'en') or role not in ('before', 'after'):
        return jsonify(error='not_found'), 404
    r = next((r for r in map(json.loads, (EVAL/split/mid/'predictions.jsonl').read_text().splitlines()) if r['case_id'] == cid and r['language'] == language), None)
    if not r or (role == 'after' and r['status'] != 'executed'):
        return jsonify(error='not_found'), 404
    p = ROOT/r['input_image' if role == 'before' else 'output_image']
    return send_from_directory(p.parent, p.name)

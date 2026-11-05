"""Actual admitted round-six data and reviewed development only; no acceptance route."""
import json,re
from flask import Blueprint,jsonify,request,send_from_directory
from functional_v1.common import ROOT,read,GPUJob,EDITOR_VERSION
from .source_bank import EXP,DATA

bp=Blueprint('functional_v6_records',__name__)
EVAL=EXP/'evaluation'/EDITOR_VERSION
MODEL=re.compile(r'(?:raw-base|current-default|round2-parent|(?:base|continued)-e[1-4])')
SPLITS=('native-development','old-native','development')

@bp.get('/api/v6/experiment')
def experiment():
    resources=read(ROOT/'experiments/functional-v1-20261104/resources.json')
    resources['occupied_gpu_hours_live']=GPUJob.consumed(resources)
    summaries={s:{p.parent.name:read(p) for p in (EVAL/s).glob('*/direct-functional-summary.json')} for s in SPLITS}
    return jsonify(admission=read(DATA/'admission.json'),
        training={p.parent.name:read(p) for p in (EXP/'runs').glob('*/status.json')},
        summaries=summaries,resources=resources,
        selection=read(EXP/'development-selection.json') if (EXP/'development-selection.json').exists() else None,
        note='AI辅助直接视觉评审；训练完成、功能合格、用户认为可用分别记录。验收未通过此接口开放。')

@bp.get('/api/v6/training-examples')
def training():
    rows=[]
    for c in read(DATA/'cases.json'):
        native=c['action']=='original-native-request'
        rows.append(dict(id=c['id'],family=c['family'],domain=c['domain'],instructions=c['instructions'],
            original_record_id=c.get('source_record_id',c['source_id']),parameters=c['parameters'],
            instruction_origin='保留原始英文／AI中文翻译' if native else 'AI构造受控任务',native=native,
            status='实际入训标签 · AI直接视觉核验（非用户认证）',
            before=f'/v6/training-image/{c["id"]}/before',after=f'/v6/training-image/{c["id"]}/after'))
    rows.sort(key=lambda r:(not r['native'],r['id']))
    return jsonify(examples=rows)

@bp.get('/v6/training-image/<cid>/<role>')
def training_image(cid,role):
    if role not in ('before','after'):return jsonify(error='not_found'),404
    c=next((c for c in read(DATA/'cases.json') if c['id']==cid),None)
    if not c:return jsonify(error='not_found'),404
    p=ROOT/c['image' if role=='before' else 'target'];return send_from_directory(p.parent,p.name)

@bp.get('/api/v6/evaluation-examples')
def examples():
    mid,split=request.args.get('model','base-e1'),request.args.get('split','native-development')
    if not MODEL.fullmatch(mid) or split not in SPLITS:return jsonify(error='not_found'),404
    folder=EVAL/split/mid
    if not (folder/'direct-review.json').exists():return jsonify(examples=[],status='visual_review_pending')
    reviews={(r['case_id'],r['language']):r['assessment'] for r in read(folder/'direct-review.json')};rows=[]
    for r in map(json.loads,(folder/'predictions.jsonl').read_text().splitlines()):
        assessment=reviews.get((r['case_id'],r['language']))
        if assessment is None:continue
        prefix=f'/v6/evaluation-image/{split}/{mid}/{r["case_id"]}/{r["language"]}'
        rows.append(dict(case_id=r['case_id'],language=r['language'],instruction=r['instruction'],model_id=mid,
            split=split,assessment=assessment,before=prefix+'/before',after=prefix+'/after' if r['status']=='executed' else None))
    rank={'major_issue':0,'minor_issue':1,'uncertain':2,'acceptable':3}
    rows.sort(key=lambda r:(rank[r['assessment']['quality']],r['assessment']['completion']=='completed',r['case_id'],r['language']))
    return jsonify(examples=rows)

@bp.get('/v6/evaluation-image/<split>/<mid>/<cid>/<language>/<role>')
def evaluation_image(split,mid,cid,language,role):
    if split not in SPLITS or not MODEL.fullmatch(mid) or language not in ('zh','en') or role not in ('before','after'):return jsonify(error='not_found'),404
    folder=EVAL/split/mid
    if not (folder/'direct-review.json').exists():return jsonify(error='not_found'),404
    row=next((r for r in map(json.loads,(folder/'predictions.jsonl').read_text().splitlines()) if r['case_id']==cid and r['language']==language),None)
    if not row or (role=='after' and row['status']!='executed'):return jsonify(error='not_found'),404
    p=ROOT/row['input_image' if role=='before' else 'output_image'];return send_from_directory(p.parent,p.name)

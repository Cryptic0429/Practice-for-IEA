"""Read-only round-2 data/results panels for the shared trial; closed acceptance stays closed."""
import json
from flask import Blueprint, jsonify, request, send_from_directory
from .common import ROOT, DATA, EXP, EVAL, SHARED_LEDGER, read

bp=Blueprint('functional_v2_records',__name__)
MODELS=('parent-v1','curriculum-e3','control-e2')

def summaries(split):
    return {p.parent.name:read(p) for p in (EVAL/split).glob('*/direct-functional-summary.json')}

@bp.get('/api/v2/experiment')
def experiment():
    return jsonify(review_mode='AI辅助直接视觉评审，无用户认证',semantic_cases=360,source_groups=60,
      training_records=720,domains={'portrait':30,'landscape':30},natural_training_requests=0,
      status='开发完成，实用目标未达成；验收未开封',
      development_results=summaries('development'),natural_results=summaries('natural-use'),
      natural_probe_note='6张开发原图的原始英文请求及中文翻译；非独立用户测试，不参与检查点选择。',
      selection=read(EXP/'development-selection.json'),resources=read(SHARED_LEDGER))

@bp.get('/api/v2/training-examples')
def training():
    rows=read(DATA/'train-final-cases.json')
    return jsonify(examples=[{'id':c['id'],'family':c['family'],'domain':c['domain'],
      'instructions':c['instructions'],'expected_change':c['expected_change'],'must_preserve':c['must_preserve'],
      'original_record_id':c['source_id'],'parameters':c['parameters'],
      'status':'实际采用标签 · AI直接看图核验；'+('人像' if c['domain']=='portrait' else '风景'),
      'before':f'/v2/training-image/{c["id"]}/before','after':f'/v2/training-image/{c["id"]}/after'} for c in rows])

@bp.get('/v2/training-image/<cid>/<role>')
def training_image(cid,role):
    if role not in ('before','after'):return jsonify(error='not_found'),404
    c=next((c for c in read(DATA/'train-final-cases.json') if c['id']==cid),None)
    if c is None:return jsonify(error='not_found'),404
    p=ROOT/c['image' if role=='before' else 'target']
    return send_from_directory(p.parent,p.name)

@bp.get('/api/v2/evaluation-examples')
def examples():
    model=request.args.get('model','curriculum-e3');split=request.args.get('split','development')
    if model not in MODELS or split not in ('development','natural-use'):
        return jsonify(error='not_found'),404
    folder=EVAL/split/model
    reviews={(r['case_id'],r['language']):r['assessment'] for r in read(folder/'direct-review.json')}
    predictions=list(map(json.loads,(folder/'predictions.jsonl').read_text().splitlines()))
    result=[]
    for r in predictions:
        assessment=reviews[r['case_id'],r['language']]
        prefix=f'/v2/evaluation-image/{split}/{model}/{r["case_id"]}/{r["language"]}'
        result.append({'case_id':r['case_id'],'language':r['language'],'family':r['family'],
            'instruction':r['instruction'],'model_id':model,'split':split,'assessment':assessment,
            'before':prefix+'/before','after':prefix+'/after' if r['status']=='executed' else None})
    # Failures first, including EN-specific damage; display both languages without counting them as separate cases.
    rank={'major_issue':0,'minor_issue':1,'uncertain':2,'acceptable':3}
    result.sort(key=lambda r:(rank[r['assessment']['quality']],r['assessment']['completion']=='completed',r['case_id'],r['language']))
    return jsonify(examples=result)

@bp.get('/v2/evaluation-image/<split>/<model>/<cid>/<language>/<role>')
def evaluation_image(split,model,cid,language,role):
    if model not in MODELS or split not in ('development','natural-use') or language not in ('zh','en') or role not in ('before','after'):
        return jsonify(error='not_found'),404
    rows=map(json.loads,(EVAL/split/model/'predictions.jsonl').read_text().splitlines())
    r=next((r for r in rows if r['case_id']==cid and r['language']==language),None)
    if r is None or (role=='after' and r['status']!='executed'):return jsonify(error='not_found'),404
    p=ROOT/r['input_image' if role=='before' else 'output_image']
    return send_from_directory(p.parent,p.name)

"""Read-only actual correction data and failed-pilot results for the trial UI."""
import json,hashlib
from flask import Blueprint,jsonify,request,send_from_directory
from .common import ROOT,DATA,EXP,EVAL,read
bp=Blueprint('functional_v3_records',__name__)
MODELS=('dpo-e1','sft-e1')

def positive_examples():
    cases={c['id']:c for c in read(DATA/'cases.json')};unique={}
    for r in read(DATA/'training-pool.json'):
        c=cases[r['case_id']];key=hashlib.sha256((c['id']+json.dumps(r['chosen'],sort_keys=True)).encode()).hexdigest()[:16]
        if key not in unique:unique[key]={'key':key,'case':c,'row':r}
    return list(unique.values())

@bp.get('/api/v3/experiment')
def experiment():
    selection=read(EXP/'development-selection.json')
    return jsonify(review_mode='AI辅助直接视觉评审，无用户认证',selection=selection,admission=read(DATA/'admission.json'),
      development_results=selection['development_results'],natural_results=selection['natural_results'],
      resources=read(ROOT/'experiments/functional-v1-20261104/resources.json'))

@bp.get('/api/v3/training-examples')
def training():
    result=[]
    for x in positive_examples():
        c,r,k=x['case'],x['row'],x['key'];is_pref=r['kind']=='preference'
        result.append({'id':c['id'],'family':c['family'],'domain':c['domain'],'instructions':c['instructions'],
           'original_record_id':c['source_id'],'parameters':r['chosen'],'instruction_origin':r['request_origin'],
           'status':'实际偏好正例 · AI直接看图' if is_pref else '实际监督保持正例 · AI直接看图',
           'before':f'/v3/training-image/{k}/before','after':f'/v3/training-image/{k}/after',
           'rejected':f'/v3/training-image/{k}/rejected' if is_pref else None})
    return jsonify(examples=result)

@bp.get('/v3/training-image/<key>/<role>')
def training_image(key,role):
    if role not in ('before','after','rejected'):return jsonify(error='not_found'),404
    x=next((x for x in positive_examples() if x['key']==key),None)
    if not x:return jsonify(error='not_found'),404
    c,r=x['case'],x['row']
    if role=='before':p=ROOT/c['image']
    elif role=='rejected':
        if r['kind']!='preference':return jsonify(error='not_found'),404
        p=ROOT/r['rejected_image']
    else:p=ROOT/(r['chosen_image'] if r['kind']=='preference' else c['target'])
    return send_from_directory(p.parent,p.name)

@bp.get('/api/v3/evaluation-examples')
def examples():
    mid=request.args.get('model','dpo-e1');split=request.args.get('split','development')
    if mid not in MODELS or split not in ('development','natural-use'):return jsonify(error='not_found'),404
    folder=EVAL/split/mid;review={(r['case_id'],r['language']):r['assessment'] for r in read(folder/'direct-review.json')};rows=list(map(json.loads,(folder/'predictions.jsonl').read_text().splitlines()));result=[]
    for r in rows:
        prefix=f'/v3/evaluation-image/{split}/{mid}/{r["case_id"]}/{r["language"]}'
        result.append({'case_id':r['case_id'],'language':r['language'],'instruction':r['instruction'],'model_id':mid,'split':split,'assessment':review[r['case_id'],r['language']],
             'before':prefix+'/before','after':prefix+'/after' if r['status']=='executed' else None})
    rank={'major_issue':0,'minor_issue':1,'uncertain':2,'acceptable':3};result.sort(key=lambda r:(rank[r['assessment']['quality']],r['assessment']['completion']=='completed',r['case_id'],r['language']))
    return jsonify(examples=result)

@bp.get('/v3/evaluation-image/<split>/<mid>/<cid>/<language>/<role>')
def evaluation_image(split,mid,cid,language,role):
    if mid not in MODELS or split not in ('development','natural-use') or language not in ('zh','en') or role not in ('before','after'):return jsonify(error='not_found'),404
    row=next((r for r in map(json.loads,(EVAL/split/mid/'predictions.jsonl').read_text().splitlines()) if r['case_id']==cid and r['language']==language),None)
    if not row or (role=='after' and row['status']!='executed'):return jsonify(error='not_found'),404
    p=ROOT/row['input_image' if role=='before' else 'output_image'];return send_from_directory(p.parent,p.name)

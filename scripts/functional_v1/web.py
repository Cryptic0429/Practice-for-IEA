"""An isolated IEA-style Flask trial; no external rewriting or translation API."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import uuid
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError
import requests
from .common import ROOT, EXP, DATA, EVAL, read, write, append, stamp
from .catalog import models, resolve

ASSETS = Path(__file__).with_name('web')
app = Flask(__name__, template_folder=str(ASSETS/'templates'), static_folder=str(ASSETS/'static'))
app.config['MAX_CONTENT_LENGTH'] = 20*1024*1024
from functional_v2.web_records import bp as round2_records
app.register_blueprint(round2_records)
from functional_v3.web_records import bp as round3_records
app.register_blueprint(round3_records)
from functional_v4.web_records import bp as round4_records
app.register_blueprint(round4_records)
from functional_v6.web_records import bp as round6_records
app.register_blueprint(round6_records)
from functional_v7.web_records import bp as round7_records
app.register_blueprint(round7_records)
from functional_v9.web_records import bp as round9_records
app.register_blueprint(round9_records)
from functional_v10.web_records import bp as round10_records
app.register_blueprint(round10_records)
from functional_v11.web_records import bp as round11_records
app.register_blueprint(round11_records)
from functional_v12.web_records import bp as round12_records
app.register_blueprint(round12_records)
from functional_v13.web_records import bp as round13_records
app.register_blueprint(round13_records)
from functional_v14.web_records import bp as round14_records
app.register_blueprint(round14_records)
from functional_v15.web_records import bp as round15_records
app.register_blueprint(round15_records)
from functional_v18.web_records import bp as round18_records
app.register_blueprint(round18_records)
from functional_v19.web_records import bp as round19_records
from functional_v20.web_records import bp as round20_records
from functional_v21.web_records import bp as round21_records
app.register_blueprint(round19_records)
app.register_blueprint(round20_records)
app.register_blueprint(round21_records)
from functional_v22.web_records import bp as current_records
app.register_blueprint(current_records)
WORKER = 'http://127.0.0.1:18282'
REQUESTS = EXP/'demo/requests'
LOCAL = re.compile(r'(?:remove|erase|delete|replace).*(?:outlet|person|people|object|background)|(?:移除|去掉|删除|替换).*(?:插座|人物|物体|背景)|(?:只|单独).*(?:脸|裙|衣服).*(?:颜色|变红|变蓝)|sharpen|锐化', re.I)


@app.get('/')
def index():
    return render_template('index.html', inference_paused=(EXP/'BUDGET_STOP.json').is_file())


@app.get('/api/models')
def get_models():
    items = models()
    public = [{k: v for k,v in item.items() if k != 'adapter'} for item in items]
    default = next((item['id'] for item in items if item.get('default')), 'current')
    return jsonify(models=public, default=default, inference_available=not (EXP/'BUDGET_STOP.json').is_file())


@app.post('/api/edit')
def edit_image():
    instruction = request.form.get('instruction', '').strip()
    model_id = request.form.get('model_id', '')
    if not instruction or 'image' not in request.files:
        return jsonify(status='failed', error='请选择图片并填写修图要求。'), 400
    try:
        spec = resolve(model_id)
    except ValueError:
        return jsonify(status='failed', error='所选模型目前不可用。'), 400
    if spec.get('runtime_family') not in ('functional_v7', 'functional_v9', 'functional_v10', 'functional_v11', 'functional_v12', 'functional_v13', 'functional_v14', 'functional_v15', 'functional_v18', 'functional_v19', 'functional_v20', 'functional_v21', 'functional_v22', 'functional_v23') and LOCAL.search(instruction):
        return jsonify(status='unsupported', error='这条要求需要局部编辑或锐化工具。当前试验版支持整体光照和调色，尚未提供这项能力。'), 422
    if (EXP/'BUDGET_STOP.json').is_file():
        return jsonify(status='failed', code='budget_exhausted', error='24 GPU 小时实验预算已结算，暂不接收新的修图请求。已保存结果可继续查看和下载。', after=None, parameters=None), 503
    request_id = uuid.uuid4().hex
    directory = REQUESTS/request_id
    directory.mkdir(parents=True, exist_ok=False)
    try:
        with Image.open(request.files['image'].stream) as im:
            if im.width*im.height > 40_000_000:
                raise ValueError('图片像素过大，请使用不超过4000万像素的图片。')
            ImageOps.exif_transpose(im).convert('RGB').save(directory/'input.png', compress_level=1)
        record = {'request_id': request_id, 'instruction': instruction, 'model_id': model_id,
                  'created_at': stamp(), 'training_use': False, 'status': 'submitted'}
        write(directory/'request.json', record)
        response = requests.post(WORKER+'/infer', json={'image': str(directory/'input.png'),
                                  'instruction': instruction, 'model_id': model_id}, timeout=180)
        result = response.json()
        if result.get('status') == 'declined':
            if spec.get('runtime_family') not in ('functional_v7', 'functional_v9', 'functional_v10', 'functional_v11', 'functional_v12', 'functional_v13', 'functional_v14', 'functional_v15', 'functional_v18', 'functional_v19', 'functional_v20', 'functional_v21', 'functional_v22', 'functional_v23'):
                raise ValueError('所选旧模型未声明拒绝输出协议。')
            if result.get('output') or result.get('output_sha256') or (directory/'output.png').exists():
                (directory/'output.png').unlink(missing_ok=True)
                raise ValueError('拒绝输出状态附带结果图，已阻止返回。')
            reason = result.get('reason_code')
            if reason not in ('unreliable_quality', 'unsupported_tools'):
                raise ValueError('模型未提供有效的无法完成原因。')
            message = ('当前工具无法在保留基本画质的同时可靠完成这条要求，因此未生成编辑结果。'
                       if reason == 'unreliable_quality' else '这条要求需要当前模型尚不支持的操作，因此未生成编辑结果。')
            record.update(status='declined', decision=result['decision'], parameters=None,
                          reason_code=reason, identity=result['identity'], completed_at=stamp())
            write(directory/'request.json', record)
            return jsonify(status='declined', request_id=request_id, before=f'/files/{request_id}/input.png',
                           after=None, parameters=None, decision=result['decision'], reason_code=reason,
                           identity=result['identity'], message=message)
        if result.get('status') != 'executed':
            record.update(status='failed', error=result.get('error', '模型推理失败。'), identity=result.get('identity'))
            write(directory/'request.json', record)
            return jsonify(status='failed', request_id=request_id,
                           error='未能生成可执行的编辑结果。', detail=record['error'], identity=record.get('identity'), after=None, parameters=None), 422
        record.update(status='executed', execution_status=result.get('execution_status', 'edited'),
                      decision=result.get('decision'), parameters=result['parameters'], identity=result['identity'],
                      output_sha256=result['output_sha256'], completed_at=stamp())
        write(directory/'request.json', record)
        return jsonify(status='executed', request_id=request_id,
                       before=f'/files/{request_id}/input.png', after=f'/files/{request_id}/output.png',
                       parameters=result['parameters'], identity=result['identity'],
                       execution_status=result.get('execution_status', 'edited'), decision=result.get('decision'),
                       message='已按要求保持原图不变。' if result.get('execution_status') == 'unchanged' else '已生成编辑结果，请检查是否符合要求。')
    except (UnidentifiedImageError, ValueError) as error:
        return jsonify(status='failed', error=str(error)), 400
    except requests.RequestException:
        return jsonify(status='failed', error='推理服务暂时不可用；请求已保留。首次加载模型需要一些时间。'), 503


@app.get('/files/<request_id>/<filename>')
def file(request_id, filename):
    if not re.fullmatch(r'[0-9a-f]{32}', request_id) or filename not in ('input.png','output.png'):
        return jsonify(error='not_found'), 404
    return send_from_directory(REQUESTS/request_id, filename)


@app.post('/api/feedback')
def feedback():
    body = request.get_json(silent=True) or {}
    request_id = body.get('request_id', '')
    verdict = body.get('verdict')
    if not re.fullmatch(r'[0-9a-f]{32}', request_id) or not (REQUESTS/request_id/'request.json').exists():
        return jsonify(error='无效的试用记录。'), 400
    if verdict not in ('completed','partial','failed','uncertain'):
        return jsonify(error='请选择符合程度。'), 400
    origin = read(REQUESTS/request_id/'request.json').get('source', 'user_trial')
    append(EXP/'demo/feedback.jsonl', {'request_id': request_id, 'verdict': verdict,
                                     'comment': str(body.get('comment',''))[:2000],
                                     'created_at': stamp(), 'source': origin, 'training_use': False})
    return jsonify(status='saved', message='反馈已保存，不会自动加入训练。')


@app.get('/api/experiment')
def experiment():
    status = {'training_started': (EXP/'runs').exists(), 'review_mode': 'AI辅助评审，无人工认证',
              'data_status': '标签候选待核验', 'training_records': 0, 'semantic_cases': 220,
              'source_groups': 160, 'languages': ['中文','English']}
    if (DATA/'label-acceptance.json').exists():
        status.update(data_status='标签已通过直接视觉核验（AI辅助）', training_records=440)
    if (EXP/'resources.json').exists():
        status['resources'] = read(EXP/'resources.json')
    summaries = {}
    directory = EVAL/'development'
    if directory.exists():
        for path in directory.glob('*/direct-functional-summary.json'):
            summaries[path.parent.name] = read(path)
    status['development_results'] = summaries
    status['acceptance_results']={p.parent.name:read(p) for p in (EVAL/'acceptance').glob('*/direct-functional-summary.json')}
    status['selection']=read(EXP/'development-selection.json') if (EXP/'development-selection.json').exists() else None
    return jsonify(status)


@app.get('/api/evaluation-examples')
def evaluation_examples():
    selection=read(EXP/'development-selection.json') if (EXP/'development-selection.json').exists() else {}
    chosen=selection.get('models',{}).get(selection.get('selected_model_id'),{}).get('evaluation_id')
    if not chosen:return jsonify(examples=[])
    split='acceptance' if (EVAL/'acceptance'/selection['selected_model_id']/'direct-review.json').exists() else 'development'
    mid=selection['selected_model_id'] if split=='acceptance' else chosen
    directory=EVAL/split/mid
    if not (directory/'direct-review.json').exists():return jsonify(examples=[])
    reviews={(r['case_id'],r['language']):r for r in read(directory/'direct-review.json')}
    predictions=[json.loads(l) for l in (directory/'predictions.jsonl').read_text().splitlines()]
    result=[]
    for row in predictions:
        if row['language']!='zh':continue
        review=reviews[(row['case_id'],'zh')]['assessment']
        result.append({'case_id':row['case_id'],'instruction':row['instruction'],'split':split,
            'model_id':mid,'assessment':review,'status':row['status'],
            'before':f'/evaluation-image/{split}/{mid}/{row["case_id"]}/before',
            'after':f'/evaluation-image/{split}/{mid}/{row["case_id"]}/after' if row['status']=='executed' else None})
    return jsonify(examples=result)


@app.get('/evaluation-image/<split>/<mid>/<cid>/<role>')
def evaluation_image(split,mid,cid,role):
    if split not in ('development','acceptance') or role not in ('before','after') or not re.fullmatch(r'[a-zA-Z0-9-]+',mid):
        return jsonify(error='not_found'),404
    path=EVAL/split/mid/'predictions.jsonl'
    if not path.exists():return jsonify(error='not_found'),404
    row=next((r for r in map(json.loads,path.read_text().splitlines()) if r['case_id']==cid and r['language']=='zh'),None)
    if not row or (role=='after' and row['status']!='executed'):return jsonify(error='not_found'),404
    image=ROOT/row['input_image' if role=='before' else 'output_image']
    return send_from_directory(image.parent,image.name)


@app.get('/api/training-examples')
def examples():
    if not (DATA/'train-cases.json').exists():
        return jsonify(examples=[])
    cases = read(DATA/'train-cases.json')
    accepted = (DATA/'label-acceptance.json').exists()
    direct = EXP/'review/training/direct-proposals.json'
    chosen = direct if direct.exists() else EXP/'curation/selections.json'
    selections = {r['case_id']: r for r in read(chosen)} if chosen.exists() else {}
    result = []
    for case in cases:
        selection = selections.get(case['id'], {}).get('selected')
        result.append({'id': case['id'], 'family': case['family'], 'instructions': case['instructions'],
                       'instruction_origin': case['instruction_origin'], 'group_id': case['group_id'],
                       'original_record_id': case['original_record_id'],
                       'before': '/training-image/'+case['id']+'/before',
                       'after': '/training-image/'+case['id']+'/after' if selection else None,
                       'parameters': selection['parameters'] if selection else None,
                       'status': ('直接视觉核验通过（AI辅助）' if accepted else '候选待逐图确认，尚未纳入训练') if selection else '候选待核验'})
    return jsonify(examples=result)


@app.get('/training-image/<case_id>/<role>')
def training_image(case_id, role):
    if not re.fullmatch(r'train-\d{3}', case_id) or role not in ('before','after'):
        return jsonify(error='not_found'), 404
    case = next((c for c in read(DATA/'train-cases.json') if c['id']==case_id), None)
    if not case:
        return jsonify(error='not_found'), 404
    path = ROOT/case['image']
    if role == 'after':
        direct = EXP/'review/training/direct-proposals.json'
        source = direct if direct.exists() else EXP/'curation/selections.json'
        if not source.exists():
            return jsonify(error='not_found'), 404
        item = next((r for r in read(source) if r['case_id']==case_id), None)
        if not item or not item.get('selected'):
            return jsonify(error='not_found'), 404
        path = ROOT/item['selected']['image']
    return send_from_directory(path.parent, path.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=12346)
    args = ap.parse_args()
    app.run(host='127.0.0.1', port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()

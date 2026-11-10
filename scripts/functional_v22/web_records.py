"""Read-only current-round evidence, split from trial uploads and sealed acceptance."""
import json,math,re
from pathlib import Path
from flask import Blueprint,abort,jsonify,render_template_string,request,send_from_directory
from functional_v1.common import ROOT,read,GPUJob
from functional_v15.outcomes import EDITOR_VERSION
bp=Blueprint('functional_current_records',__name__)
SPLITS=('native-development','old-native','development')
def paths(v):
 if v not in (22,23):abort(404)
 return ROOT/f'experiments/functional-v{v}-20261107',ROOT/'data/training/functional-v22-20261107'
def overview(v):
 ex,da=paths(v);p=read(ex/'training-protocol.json');ledger=read(ROOT/'experiments/functional-v1-20261104/resources.json')
 return dict(version=v,data=read(da/'admission.json')['counts'],data_sha256=__import__('hashlib').sha256((da/'cases.json').read_bytes()).hexdigest(),occupied_gpu_hours=GPUJob.consumed(ledger),budget=ledger['limit_gpu_hours'],
  training={f.parent.name:read(f) for f in (ex/'runs').glob('*/status.json')},coverage={f.parent.name:read(f) for f in (ex/'runs').glob('*/actual-data-coverage.json')},
  summaries={s:{f.parent.name:read(f) for f in (ex/'evaluation'/EDITOR_VERSION/s).glob('*/direct-functional-summary.json') if read(f).get('status')=='completed'} for s in SPLITS},
  selection=read(ex/'development-selection.json') if (ex/'development-selection.json').exists() else None,acceptance_exposed=False,independent_user_feedback=0)
@bp.get('/api/v<int:v>/experiment')
def api(v):return jsonify(overview(v))
def records(v,kind,model,split):
 ex,da=paths(v);rows=[]
 if kind=='training':
  for c in read(da/'cases.json'):
   prefix=f'/v{v}/training-image/{c["id"]}'
   rows.append(dict(id=c['id'],domain=c['domain'],action=c['action'],instructions=c['instructions'],origin=c['instruction_origin'],parameters=c.get('parameters'),judgment='label',status='已核验训练标签；不是模型结果',evidence=json.dumps(c.get('label_evidence',{}),ensure_ascii=False),before=prefix+'/before',after=prefix+'/after' if c.get('target') else None,no_result='训练标签：保持质量无法可靠完成或工具不支持，明确拒绝，不返回结果。'))
 elif kind=='evaluation':
  folder=ex/'evaluation'/EDITOR_VERSION/split/model
  if not (folder/'direct-review.json').exists():return rows
  notes={(r['case_id'],r['language']):r['assessment'] for r in read(folder/'direct-review.json')}
  for r in map(json.loads,(folder/'predictions.jsonl').read_text().splitlines()):
   n=notes.get((r['case_id'],r['language']))
   if not n:continue
   judgment='pass' if n['completion']=='completed' and n['quality']=='acceptable' else 'fail' if n['completion'] in ('partial','failed') or n['quality'] in ('minor_issue','major_issue') else 'uncertain'
   prefix=f'/v{v}/evaluation-image/{split}/{model}/{r["case_id"]}/{r["language"]}'
   rows.append(dict(id=r['case_id']+'-'+r['language'],domain=r['domain'],action=r['status'],instructions={r['language']:r['instruction']},origin=model,parameters=r.get('parameters'),judgment=judgment,status=n['completion']+' / '+n['quality'],evidence=n['evidence'],before=prefix+'/before',after=prefix+'/after' if r['status']=='executed' else None,no_result='未返回结果图，原修图要求不计为完成；拒绝质量另评。'))
 return rows
@bp.get('/v<int:v>')
def page(v):
 info=overview(v);kind=request.args.get('kind','training');split=request.args.get('split','native-development');model=request.args.get('model','');domain=request.args.get('domain','all');judgment=request.args.get('judgment','all')
 if kind not in ('training','evaluation') or split not in SPLITS or model and not re.fullmatch('[a-z0-9-]+',model):abort(404)
 mids=sorted(info['summaries'][split]);model=model or (mids[0] if mids else '')
 rows=[r for r in records(v,kind,model,split) if (domain=='all' or domain==r['domain']) and (judgment=='all' or judgment==r['judgment'] or kind=='training')]
 page=max(1,min(request.args.get('page',1,type=int),max(1,math.ceil(len(rows)/12))))
 return render_template_string(Path(__file__).with_name('records.html').read_text(),**info,rows=rows[(page-1)*12:page*12],kind=kind,split=split,model=model,model_ids=mids,domain=domain,judgment=judgment,total=len(rows),page=page,pages=max(1,math.ceil(len(rows)/12)))
@bp.get('/v<int:v>/training-image/<cid>/<role>')
def training_image(v,cid,role):
 ex,da=paths(v);c=next((c for c in read(da/'cases.json') if c['id']==cid),None)
 if not c or role not in ('before','after') or role=='after' and not c.get('target'):abort(404)
 f=ROOT/c['image' if role=='before' else 'target'];return send_from_directory(f.parent,f.name)
@bp.get('/v<int:v>/evaluation-image/<split>/<model>/<cid>/<language>/<role>')
def eval_image(v,split,model,cid,language,role):
 ex,da=paths(v)
 if split not in SPLITS or not re.fullmatch('[a-z0-9-]+',model) or language not in ('zh','en') or role not in ('before','after'):abort(404)
 folder=ex/'evaluation'/EDITOR_VERSION/split/model
 if not (folder/'direct-review.json').exists():abort(404)
 r=next((r for r in map(json.loads,(folder/'predictions.jsonl').read_text().splitlines()) if r['case_id']==cid and r['language']==language),None)
 if not r or role=='after' and r['status']!='executed':abort(404)
 f=ROOT/r['input_image' if role=='before' else 'output_image'];return send_from_directory(f.parent,f.name)

"""Eight actual requests, no cached inference substituted for a model call."""
import json,requests,hashlib
import numpy as np
from .common import *
from functional_v1.common import EXP as BUDGET
from functional_v1.editor import load_rgb

def main():
 p=protocol();url='http://127.0.0.1:12346';selection=read(EXP/'development-selection.json');records=[]
 controls=read(ROOT/p['sources']['decision_development']['path']);native=read(ROOT/p['sources']['native_development']['path']);prior=read(ROOT/'experiments/functional-v19-20261107/demo/case12-protocol.json')
 tasks=[('historical-case12-not-independent',prior,'en'),('native-development',next(c for c in native if c['id']=='native-dev-070'),'zh')]+[('decision-development',next(c for c in controls if c['expected_action']==a),'en') for a in ('keep','decline')]
 write(EXP/'demo/trial-protocol.json',dict(at=stamp(),fixed_models={k:m['adapter_sha256'] for k,m in selection['models'].items()},tasks=[dict(split=s,case_id=c['id'],language=l,input_sha256=sha(ROOT/c['image']),instruction=c['instructions'][l]) for s,c,l in tasks],purpose='Live literal offline-runtime identity and no-result/keep contract, not independent functional scoring.',independent_user_feedback=0))
 try:
  for mid,m in selection['models'].items():
   eid=m['evaluation_id'];offline={}
   for split in ('native-development','decision-development'):
    path=EXP/'evaluation/functional-editor-v15-shared-shadow-gain-candidate'/split/eid/'generations.jsonl'
    if path.exists():offline[split]={(r['case_id'],r['language']):r for r in map(json.loads,path.read_text().splitlines())}
   for split,c,l in tasks:
    with (ROOT/c['image']).open('rb') as f:r=requests.post(url+'/api/edit',files={'image':('source.png',f,'image/png')},data=dict(model_id=mid,instruction=c['instructions'][l]),timeout=220)
    a=r.json();rid=a.get('request_id');assert rid,a;saved=BUDGET/'demo/requests'/rid/'request.json';request=read(saved);request['source']='agent_round23_HTTP_test_not_user_feedback';write(saved,request)
    assert not request['training_use'] and request['instruction']==c['instructions'][l]
    assert r.status_code==200 and a['status'] in ('executed','declined'),a
    identity=a['identity'];assert identity['runtime_family']=='functional_v23' and identity['adapter_sha256']==m['adapter_sha256'] and identity['parent_adapter_sha256']==PARENT_SHA and identity['training_data_sha256']==p['data_files']['cases.json']
    expected=offline.get(split,{}).get((c['id'],l));parity=None
    if expected:parity=a['decision']==expected['decision'];assert parity,(a,expected)
    if a['status']=='declined':assert a['after'] is None and a['parameters'] is None and not saved.with_name('output.png').exists() and requests.get(url+f'/files/{rid}/output.png',timeout=10).status_code==404
    else:
     output=requests.get(url+a['after'],timeout=15);assert output.status_code==200
     if a['decision']['action']=='keep':assert np.array_equal(load_rgb(saved.with_name('input.png')),load_rgb(saved.with_name('output.png')))
    records.append(dict(request_id=rid,model_id=mid,case_id=c['id'],language=l,split=split,status=a['status'],decision=a['decision'],offline_decision_parity=parity,adapter_sha256=identity['adapter_sha256'],output_sha256=sha(saved.with_name('output.png')) if a['status']=='executed' else None))
    write(EXP/'demo/http-progress.json',dict(at=stamp(),checks=records,independent_user_feedback=0))
  write(EXP/'demo/http-integration.json',dict(at=stamp(),status='passed_actual_new_LoRA_runtime_HTTP',checks=records,count=len(records),default_unchanged=requests.get(url+'/api/models',timeout=10).json()['default']=='functional-current-20261104',training_use=False,independent_user_feedback=0,quality_score=False))
 finally:requests.post('http://127.0.0.1:18282/unload',timeout=45).raise_for_status()
 print('Eight new-LoRA live HTTP requests passed, not a quality verdict')
if __name__=='__main__':main()

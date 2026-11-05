"""Actual HTTP edits: verify adapters, bilingual/native-image parity, download, feedback isolation."""
import json
import hashlib
import requests
from .common import ROOT, DATA, EXP, EVAL, read, write, stamp


def main():
    base='http://127.0.0.1:12346'
    models=requests.get(base+'/api/models').json()
    assert models['default']=='functional-current-20261104'
    selection=read(EXP/'development-selection.json')
    assert set(selection['models']) <= {m['id'] for m in models['models']}
    checks=[]
    for route,cid in [('curriculum','natural-028'),('control','natural-033')]:
        mid=f'functional-v2-{route}-20261106';spec=selection['models'][mid]
        eid=spec['evaluation_id'];rows=[json.loads(l) for l in (EVAL/'natural-use'/eid/'predictions.jsonl').read_text().splitlines()]
        for language in ('zh','en'):
            expected=next(r for r in rows if r['case_id']==cid and r['language']==language)
            with (ROOT/expected['input_image']).open('rb') as f:
                response=requests.post(base+'/api/edit',files={'image':('native.png',f,'image/png')},
                    data={'instruction':expected['instruction'],'model_id':mid},timeout=200)
            assert response.status_code==200,response.text
            result=response.json();request_id=result['request_id']
            # Agent checks are not user trial feedback and never become training records.
            record_path=ROOT/'experiments/functional-v1-20261104/demo/requests'/request_id/'request.json'
            record=read(record_path);record['source']='agent_smoke_test_not_user_feedback';write(record_path,record)
            assert result['identity']['adapter_sha256']==spec['adapter_sha256']
            assert result['parameters']==expected['parameters']
            assert result['identity']['runtime_sha256']==read(EVAL/'natural-use'/eid/'identity.json')['runtime_sha256']
            download=requests.get(base+result['after']);assert download.status_code==200
            assert hashlib.sha256(download.content).hexdigest()==expected['output_sha256']
            if route=='curriculum' and language=='en':
                fb=requests.post(base+'/api/feedback',json={'request_id':request_id,'verdict':'failed','comment':'Agent integration check: preserve the known English overexposure failure; not user feedback.'})
                assert fb.status_code==200
            checks.append({'model_id':mid,'case_id':cid,'language':language,'request_id':request_id,
                'adapter_sha256':spec['adapter_sha256'],'parameters':result['parameters'],
                'download_matches_offline_bytes':True,'source':'agent_smoke_test_not_user_feedback'})
            print(json.dumps(checks[-1]),flush=True)
    assert requests.get(base+'/api/v2/experiment').json()['selection']['acceptance_status']=='closed_not_evaluated'
    assert requests.get(base+'/api/v2/evaluation-examples?split=acceptance').status_code==404
    assert requests.get(base+'/v2/training-image/bad/after').status_code==404
    assert requests.get(base+'/api/v2/evaluation-examples?model=unknown').status_code==404
    data=requests.get(base+'/api/v2/training-examples').json();assert len(data['examples'])==360
    assert requests.get(base+data['examples'][0]['after']).status_code==200
    assert requests.post(base+'/api/edit',data={'model_id':'unknown','instruction':'brighten'},files={'image':('x.png',b'invalid')}).status_code==400
    requests.post('http://127.0.0.1:18282/unload')
    write(EXP/'demo/http-integration.json',{'status':'passed','at':stamp(),'checks':checks,
       'human_feedback':False,'acceptance_opened':False,'default_unchanged':True,
       'security_checks':'Unregistered model, nonexistent case, closed acceptance and malformed edit rejected.',
       'data_images':'360 actual targets available; first native target downloaded successfully'})

if __name__=='__main__':main()

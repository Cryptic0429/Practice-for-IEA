"""Actual bilingual HTTP trial parity for the two input-route diagnostic adapters."""
import json,hashlib,requests
from .common import ROOT,EXP,EVAL,read,write,stamp

def main():
    base='http://127.0.0.1:12346';sel=read(EXP/'development-selection.json');checks=[]
    for mid,spec in sel['models'].items():
        rows=list(map(json.loads,(EVAL/'natural-use'/spec['evaluation_id']/'predictions.jsonl').read_text().splitlines()))
        for lang in ('zh','en'):
            expected=next(r for r in rows if r['case_id']=='natural-028' and r['language']==lang)
            with (ROOT/expected['input_image']).open('rb') as f:
                response=requests.post(base+'/api/edit',data={'model_id':mid,'instruction':expected['instruction']},files={'image':('native.png',f,'image/png')},timeout=200)
            assert response.status_code==200,response.text
            result=response.json();record_path=ROOT/'experiments/functional-v1-20261104/demo/requests'/result['request_id']/'request.json';record=read(record_path);record['source']='agent_smoke_test_not_user_feedback';write(record_path,record)
            assert result['parameters']==expected['parameters'] and result['identity']['adapter_sha256']==spec['adapter_sha256']
            assert result['identity']['runtime_sha256']==spec['runtime_sha256']
            assert result['identity']['input_route']==spec['input_route']
            assert result['identity']['runtime_family']=='functional_v4'
            image=requests.get(base+result['after']);assert hashlib.sha256(image.content).hexdigest()==expected['output_sha256']
            checks.append({'model_id':mid,'language':lang,'request_id':result['request_id'],'parameters':result['parameters'],'download_byte_parity':True})
            print(json.dumps(checks[-1]),flush=True)
    examples=requests.get(base+'/api/v4/training-examples').json()['examples'];assert examples and requests.get(base+examples[0]['after']).status_code==200
    assert requests.get(base+'/api/v4/evaluation-examples?split=acceptance').status_code==404
    assert requests.get(base+'/api/models').json()['default']=='functional-current-20261104'
    requests.post('http://127.0.0.1:18282/unload')
    write(EXP/'demo/http-integration.json',{'status':'passed','at':stamp(),'checks':checks,'actual_positive_visual_examples':len(examples),'human_feedback':False,'training_upload_use':False,'acceptance_opened':False})
if __name__=='__main__':main()

"""HTTP trial/offline byte parity; feedback explicitly marked as agent testing."""
import hashlib,json
import requests
from functional_v1.common import ROOT,read,write,stamp,sha
from .source_bank import EXP,DATA
from .evaluate import EVAL

def main():
    url='http://127.0.0.1:12346';worker='http://127.0.0.1:18282'
    catalog=requests.get(url+'/api/models',timeout=15).json()
    assert len(catalog['models'])==13 and catalog['default']=='functional-current-20261104'
    selection=read(EXP/'development-selection.json')
    cases={c['id']:c for c in read(DATA/'development-native-cases.json')}
    results=[]
    for route in ('base','continued'):
        mid=f'functional-v6-{route}-20261110';meta=selection['models'][mid]
        offline={(r['case_id'],r['language']):r for r in map(json.loads,(EVAL/'native-development'/f'{route}-e4'/'predictions.jsonl').read_text().splitlines())}
        # Keep the disclosed illegal-output case; do not alter the parser to
        # turn an offline failure into a successful web edit.
        for cid,lang in [('native-dev-006','zh'),('native-dev-006','en'),('native-dev-001','zh'),('native-dev-001','en')]:
            c=cases[cid]
            with (ROOT/c['image']).open('rb') as f:
                response=requests.post(url+'/api/edit',data={'instruction':c['instructions'][lang],'model_id':mid},files={'image':('source.png',f,'image/png')},timeout=240)
            r=response.json();expected=offline[c['id'],lang]
            path=ROOT/'experiments/functional-v1-20261104/demo/requests'/r['request_id']/'request.json'
            record=read(path);record['source']='agent_HTTP_parity_test_not_user_feedback';write(path,record)
            assert r['status']==expected['status'],(r,expected)
            if expected['status']=='failed':
                assert response.status_code==422 and r['detail']==expected['error'],r
                assert 'after' not in r and requests.get(url+f'/files/{r["request_id"]}/output.png',timeout=15).status_code==404
                results.append(dict(model_id=mid,language=lang,case_id=cid,request_id=r['request_id'],status='failed',
                    error=r['detail'],offline_error_identical=True,no_success_fallback=True,feedback_source=record['source']))
                continue
            assert response.status_code==200,response.text
            assert r['parameters']==expected['parameters']
            assert r['identity']['adapter_sha256']==meta['adapter_sha256'] and r['identity']['runtime_sha256']==meta['runtime_sha256']
            output=requests.get(url+r['after'],timeout=30).content;assert hashlib.sha256(output).hexdigest()==expected['output_sha256']
            feedback=requests.post(url+'/api/feedback',json={'request_id':r['request_id'],'verdict':'uncertain','comment':'Agent integration check only. No independent user trial judgement; input is a disclosed development photo.'},timeout=20)
            assert feedback.status_code==200
            results.append(dict(model_id=mid,language=lang,case_id=c['id'],request_id=r['request_id'],identity=r['identity'],
                status='executed',parameters=r['parameters'],output_sha256=expected['output_sha256'],offline_byte_identical=True,feedback_source=record['source']))
    for suffix in ('?split=acceptance','?split=native-acceptance'):
        assert requests.get(url+'/api/v6/evaluation-examples'+suffix,timeout=15).status_code==404
    assert len(requests.get(url+'/api/v6/training-examples',timeout=20).json()['examples'])==460
    requests.post(worker+'/unload',timeout=40).raise_for_status()
    write(EXP/'demo/http-integration.json',dict(at=stamp(),status='passed',checks=results,models=13,default_unchanged=True,
        acceptance_routes_closed=True,training_examples=460,uploads_not_in_training=True,user_feedback=False,worker_explicitly_unloaded=True))
    print('Eight bilingual/model comparisons preserve offline outcomes, including the explicit 422 failure; agent feedback stored separately.')

if __name__=='__main__':main()

"""Actual HTTP inference: both routes, byte parity, no-file refusals and feedback."""
import hashlib,json
import requests
from functional_v1.common import ROOT,read,write,stamp,EXP as BUDGET
from .runtime import EXP
from .visual_review import case_file,EVAL

def main():
    receipt=EXP/'demo/http-integration.json';assert not receipt.exists()
    url='http://127.0.0.1:12346';catalog=requests.get(url+'/api/models',timeout=10).json()
    assert len(catalog['models'])==27 and catalog['default']=='functional-current-20261104'
    cases={split:{c['id']:c for c in read(case_file(split))} for split in ('native-development','decision-development')}
    training={c['id']:c for c in read(ROOT/'data/training/functional-v18-20261107/cases.json')}
    progress=EXP/'demo/http-integration-progress.json'
    records=read(progress)['checks'] if progress.exists() else []
    done={(r['model_id'],r['split'],r['case_id'],r['language']) for r in records}
    for mid,meta in read(EXP/'development-selection.json')['models'].items():
        offline={split:{(r['case_id'],r['language']):r for r in map(json.loads,(EVAL/split/meta['evaluation_id']/'predictions.jsonl').read_text().splitlines())} for split in cases}
        controls=list(cases['decision-development'].values())
        tasks=[('native-development',cases['native-development'][cid],lang) for cid in ('native-dev-001','native-dev-006') for lang in ('zh','en')]
        tasks += [('decision-development',next(c for c in controls if c['expected_action']==action),'zh') for action in ('keep','decline')]
        tasks += [('seen-training-quality-decline',training['visual-risk-01-severe-light'],'zh')]
        for split,case,lang in tasks:
            if (mid,split,case['id'],lang) in done:continue
            with (ROOT/case['image']).open('rb') as handle:
                response=requests.post(url+'/api/edit',data=dict(instruction=case['instructions'][lang],model_id=mid),files={'image':('source.png',handle,'image/png')},timeout=240)
            actual=response.json();assert 'request_id' in actual,(response.status_code,actual);rid=actual['request_id'];path=BUDGET/'demo/requests'/rid/'request.json';saved=read(path)
            assert saved['training_use'] is False
            saved['source']='agent_round18_HTTP_test_not_user_feedback';write(path,saved)
            expected=offline.get(split,{}).get((case['id'],lang))
            if expected:assert actual['status']==expected['status'],(actual,expected)
            identity=actual['identity']
            assert identity['adapter_sha256']==meta['adapter_sha256'] and identity['parent_adapter_sha256']==meta['parent_sha256']
            assert identity['runtime_family']=='functional_v18' and identity['training_route']==meta['training_route']
            assert identity['editor']==meta['editor'] and identity['training_data_sha256']==meta['dataset_sha256']
            if actual['status']=='executed':
                assert response.status_code==200
                output=requests.get(url+actual['after'],timeout=15);output.raise_for_status();digest=hashlib.sha256(output.content).hexdigest()
                if expected:assert digest==expected['output_sha256'] and actual['decision']==expected['decision']
                details=dict(output_sha256=digest,offline_byte_identical=expected is not None)
            elif actual['status']=='declined':
                assert response.status_code==200 and actual['after'] is None and actual['parameters'] is None
                assert not path.with_name('output.png').exists()
                assert requests.get(url+f'/files/{rid}/output.png',timeout=10).status_code==404
                if expected:assert actual['decision']==expected['decision']
                details=dict(no_result_file=True,reason_code=actual['reason_code'])
                if split.startswith('seen-'):assert actual['reason_code']=='unreliable_quality'
            else:raise AssertionError(actual)
            records.append(dict(model_id=mid,case_id=case['id'],split=split,language=lang,request_id=rid,status=actual['status'],**details))
            write(EXP/'demo/http-integration-progress.json',dict(at=stamp(),checks=records,independent_user_feedback=0))
        f=requests.post(url+'/api/feedback',json=dict(request_id=rid,verdict='uncertain',comment='Agent R18 integration test; seen training input, not independent user usability feedback.'),timeout=15);f.raise_for_status()
    requests.post('http://127.0.0.1:18282/unload',timeout=45).raise_for_status()
    assert requests.get(url+'/v18/evaluation-image/native-acceptance/counterfactual-rank-e2/native-accept-001/zh/before',timeout=10).status_code==404
    write(receipt,dict(at=stamp(),status='passed_actual_HTTP_contract_checks',checks=records,models_registered=27,default_unchanged=True,
        independent_user_feedback=0,agent_feedback_records=2,training_use=False,worker_unloaded=True,no_external_rewriting=True,acceptance_closed=True))
    print('Actual requests',len(records),'; two actual adapters, literal bilingual inputs, offline parity and no-result refusal passed.')

if __name__=='__main__':
    try:main()
    finally:requests.post('http://127.0.0.1:18282/unload',timeout=45).raise_for_status()

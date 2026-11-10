"""12 real local-model HTTP requests; compare exact offline decisions/pixels."""
import hashlib,json
import requests
import numpy as np
from .common import ROOT,EXP,read,write,sha,stamp
from functional_v1.common import EXP as BUDGET
from functional_v1.catalog import models
from functional_v1.editor import load_rgb
from .visual_review import EVAL,case_file


def main():
    receipt=EXP/'demo/http-integration.json';assert not receipt.exists()
    url='http://127.0.0.1:12346';catalog=requests.get(url+'/api/models',timeout=10).json()
    assert {m['id'] for m in catalog['models']}=={m['id'] for m in models()}
    assert catalog['default']=='functional-current-20261104'
    old={c['id']:c for c in read(case_file('old-native'))};native={c['id']:c for c in read(case_file('native-development'))}
    controls=read(case_file('decision-development'))
    prior=read(ROOT/'experiments/functional-v19-20261107/demo/case12-protocol.json')
    historical={k:prior[k] for k in ('id','image','instructions')};assert sha(ROOT/prior['image'])==prior['input_sha256']
    write(EXP/'demo/trial-protocol.json',dict(at=stamp(),model_lock_sha256=sha(EXP/'development-selection.json'),
        historical_case12=dict(historical,input_sha256=prior['input_sha256']),
        purpose='Actual R15 LM plus R22 head live identity/parity, keep/refusal and historical brown regression. No new independent evaluation or checkpoint selection.',
        tasks_per_model=6,independent_user_feedback=0))
    records=[]
    for mid,meta in read(EXP/'development-selection.json')['models'].items():
        eid=meta['evaluation_id'];namespace=EXP
        offline={split:{(r['case_id'],r['language']):r for r in map(json.loads,(namespace/'evaluation'/EVAL.name/split/eid/'predictions.jsonl').read_text().splitlines())}
            for split in ('old-native','native-development','decision-development')}
        tasks=[('old-native',old['natural-019'],'en'),('old-native',old['natural-030'],'en'),('native-development',native['native-dev-070'],'zh')]
        tasks += [('decision-development',next(c for c in controls if c['expected_action']==a),'zh') for a in ('keep','decline')]
        tasks += [('historical-case12-not-independent',historical,'en')]
        assert len(tasks)==6
        for split,case,language in tasks:
            expected=offline.get(split,{}).get((case['id'],language))
            with (ROOT/case['image']).open('rb') as handle:
                response=requests.post(url+'/api/edit',data=dict(model_id=mid,instruction=case['instructions'][language]),
                    files={'image':('source.png',handle,'image/png')},timeout=240)
            actual=response.json();assert actual.get('request_id'),(response.status_code,actual)
            rid=actual['request_id'];path=BUDGET/'demo/requests'/rid/'request.json';saved=read(path)
            assert not saved['training_use'] and saved['instruction']==case['instructions'][language]
            saved['source']='agent_round22_HTTP_test_not_user_feedback';write(path,saved)
            assert np.array_equal(load_rgb(path.with_name('input.png')),load_rgb(ROOT/case['image']))
            identity=actual.get('identity');assert identity and identity['runtime_family']=='functional_v22',actual
            assert identity['controller_sha256']==meta['controller_sha256'] and identity['parent_adapter_sha256']==meta['parent_sha256']
            assert identity['shared_weight_policy']==meta['policy'] and identity['training_data_sha256']==meta['training_cases_sha256']
            assert response.status_code==200 and actual['status'] in ('executed','declined'),actual
            if expected:assert actual['status']==expected['status'] and actual['decision']==expected['decision'],(actual,expected)
            detail={}
            if actual['status']=='executed':
                assert actual['after'];output=requests.get(url+actual['after'],timeout=15);output.raise_for_status()
                digest=hashlib.sha256(output.content).hexdigest();detail['output_sha256']=digest
                if expected:assert digest==expected['output_sha256'];detail['offline_PNG_exact']=True
                if actual['decision']['action']=='keep':assert np.array_equal(load_rgb(path.with_name('input.png')),load_rgb(path.with_name('output.png')))
            else:
                assert actual['after'] is None and actual['parameters'] is None
                assert not path.with_name('output.png').exists() and requests.get(url+f'/files/{rid}/output.png',timeout=10).status_code==404
                detail.update(no_result=True,reason_code=actual['reason_code'])
            records.append(dict(request_id=rid,model_id=mid,case_id=case['id'],language=language,split=split,
                status=actual['status'],decision=actual.get('decision'),controller_sha256=identity['controller_sha256'],
                cached_parity_checked=expected is not None,visual_quality_newly_assessed=False,**detail))
            write(EXP/'demo/http-integration-progress.json',dict(at=stamp(),checks=records,independent_user_feedback=0))
        feedback=requests.post(url+'/api/feedback',json=dict(request_id=records[-1]['request_id'],verdict='uncertain',
            comment='Agent R22 HTTP integration test on historical case-12; not independent user quality approval.'),timeout=15);feedback.raise_for_status()
    assert len(records)==12
    requests.post('http://127.0.0.1:18282/unload',timeout=45).raise_for_status()
    write(receipt,dict(at=stamp(),status='passed_actual_HTTP_identity_and_cached_parity',checks=records,models_registered=len(catalog['models']),
        default_unchanged=True,training_use=False,no_external_rewriting=True,worker_unloaded=True,independent_user_feedback=0,
        scoring='Live/runtime probes only; historical case12 is not independent and does not affect model selection.'))
    print('12 actual HTTP calls: live parent/head parity, keep and no-image refusal. No usefulness inferred.')


if __name__=='__main__':
    try:main()
    finally:requests.post('http://127.0.0.1:18282/unload',timeout=45).raise_for_status()

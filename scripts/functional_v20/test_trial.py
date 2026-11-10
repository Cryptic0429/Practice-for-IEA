"""Actual HTTP inference against fixed failed heads; cached parity, no quality score."""
import hashlib,json
import requests
import numpy as np
import torch
from .common import ROOT,EXP,DATA,read,write,sha,stamp
from .controller import load_controller,predict_cached
from functional_v1.common import EXP as BUDGET
from functional_v1.catalog import models
from functional_v1.editor import load_rgb
from .visual_review import EVAL,case_file

def main():
    receipt=EXP/'demo/http-integration.json';assert not receipt.exists()
    url='http://127.0.0.1:12346';catalog=requests.get(url+'/api/models',timeout=10).json()
    assert {m['id'] for m in catalog['models']}=={m['id'] for m in models()}
    assert catalog['default']=='functional-current-20261104'
    old={c['id']:c for c in read(case_file('old-native'))}
    native={c['id']:c for c in read(case_file('native-development'))}
    controls=read(case_file('decision-development'))
    # Historical photo was already repeatedly viewed, frozen before these probes,
    # after model/checkpoint failure registration; never used to select a head.
    prior=read(ROOT/'experiments/functional-v19-20261107/demo/case12-protocol.json')
    historical={k:prior[k] for k in ('id','image','instructions')}
    assert sha(ROOT/prior['image'])==prior['input_sha256']
    write(EXP/'demo/trial-protocol.json',dict(at=stamp(),model_lock_sha256=sha(EXP/'development-selection.json'),
        old_cases=['natural-028','natural-019','natural-030'],historical_case12=dict(historical,input_sha256=prior['input_sha256']),
        purpose='Runtime identity/parity, actual error/decline/keep and historical brown probe. Not checkpoint selection or new independent quality assessment.',
        extra_mean_controls='Only2 runtime probes; not a20-case functional control score.',independent_user_feedback=0))
    records=[]
    for mid,meta in read(EXP/'development-selection.json')['models'].items():
        eid=meta['evaluation_id'];net,item=load_controller(ROOT/meta['controller'])
        offline={}
        for split in ('old-native','native-development','decision-development'):
            path=EVAL/split/eid/'predictions.jsonl'
            if path.exists():offline[split]={(r['case_id'],r['language']):r for r in map(json.loads,path.read_text().splitlines())}
        tasks=[('old-native',old[cid],l) for cid in ('natural-028','natural-019','natural-030') for l in ('zh','en')]
        tasks += [('decision-development',next(c for c in controls if c['expected_action']==a),'zh') for a in ('keep','decline')]
        if meta['route']=='last':
            row=next(r for r in offline['native-development'].values() if r['case_id']=='native-dev-070' and r['status']=='declined')
            tasks += [('native-development',native['native-dev-070'],row['language'])]
            tasks += [('historical-case12-not-independent',historical,l) for l in ('zh','en')]
        else:
            row=next(r for r in offline['old-native'].values() if r['status']=='failed')
            tasks += [('old-native',old[row['case_id']],row['language'])]
            tasks += [('historical-case12-not-independent',historical,'en')]
        control_cache={(r['case_id'],r['language']):r for r in read(EXP/'features/decision_development-manifest.json')['rows']}
        for split,case,language in tasks:
            expected=offline.get(split,{}).get((case['id'],language))
            if expected is None and split=='decision-development':
                cached=control_cache[case['id'],language]
                assert sha(ROOT/cached['path'])==cached['sha256']
                _,decision=predict_cached(net,item,torch.load(ROOT/cached['path'],map_location='cpu',weights_only=True))
                expected=dict(status='declined' if decision['action']=='decline' else 'executed',decision=decision)
            with (ROOT/case['image']).open('rb') as handle:
                response=requests.post(url+'/api/edit',data=dict(model_id=mid,instruction=case['instructions'][language]),
                    files={'image':('source.png',handle,'image/png')},timeout=240)
            actual=response.json();assert actual.get('request_id'),(response.status_code,actual)
            rid=actual['request_id'];path=BUDGET/'demo/requests'/rid/'request.json';saved=read(path)
            assert not saved['training_use'] and saved['instruction']==case['instructions'][language]
            saved['source']='agent_round20_HTTP_test_not_user_feedback';write(path,saved)
            assert np.array_equal(load_rgb(path.with_name('input.png')),load_rgb(ROOT/case['image']))
            identity=actual.get('identity');assert identity and identity['runtime_family']=='functional_v20',actual
            assert identity['controller_sha256']==meta['controller_sha256'] and identity['parent_adapter_sha256']==meta['parent_sha256']
            assert identity['pooling']==meta['route'] and identity['training_data_sha256']==meta['training_cases_sha256']
            if expected:assert actual['status']==expected['status'],(actual,expected)
            detail={}
            if actual['status']=='executed':
                assert response.status_code==200 and actual['after']
                output=requests.get(url+actual['after'],timeout=15);output.raise_for_status()
                digest=hashlib.sha256(output.content).hexdigest();detail['output_sha256']=digest
                if expected:
                    assert actual['decision']==expected['decision']
                    if expected.get('output_sha256'):assert digest==expected['output_sha256'];detail['offline_PNG_exact']=True
                if actual['decision']['action']=='keep':assert np.array_equal(load_rgb(path.with_name('input.png')),load_rgb(path.with_name('output.png')))
            elif actual['status']=='declined':
                assert response.status_code==200 and actual['after'] is None and actual['parameters'] is None
                assert not path.with_name('output.png').exists() and requests.get(url+f'/files/{rid}/output.png',timeout=10).status_code==404
                if expected:assert actual['decision']==expected['decision']
                detail.update(no_result=True,reason_code=actual['reason_code'])
            else:
                assert response.status_code==422 and actual['after'] is None and actual['parameters'] is None
                assert not path.with_name('output.png').exists() and requests.get(url+f'/files/{rid}/output.png',timeout=10).status_code==404
                assert expected and actual['detail']==expected['error']
                detail.update(no_result=True,offline_error_exact=True)
            records.append(dict(request_id=rid,model_id=mid,case_id=case['id'],language=language,split=split,
                status=actual['status'],decision=actual.get('decision'),controller_sha256=identity['controller_sha256'],
                cached_parity_checked=expected is not None,visual_quality_newly_assessed=False,**detail))
            write(EXP/'demo/http-integration-progress.json',dict(at=stamp(),checks=records,independent_user_feedback=0))
        last=records[-1]
        feedback=requests.post(url+'/api/feedback',json=dict(request_id=last['request_id'],verdict='uncertain',
            comment='Agent R20 HTTP integration test on historical case-12; no independent user quality approval.'),timeout=15);feedback.raise_for_status()
    assert len(records)==21
    requests.post('http://127.0.0.1:18282/unload',timeout=45).raise_for_status()
    assert not torch.cuda.is_initialized()
    write(receipt,dict(at=stamp(),status='passed_actual_HTTP_identity_and_cached_parity',checks=records,models_registered=len(catalog['models']),
        default_unchanged=True,training_use=False,no_external_rewriting=True,worker_unloaded=True,independent_user_feedback=0,
        scoring='Protocol/runtime probes only; no new native score for stopped mean route andno historical-independent claim.'))
    print('21 actual HTTP requests; image/decision/error parity, no quality score inferred.')

if __name__=='__main__':
    try:main()
    finally:requests.post('http://127.0.0.1:18282/unload',timeout=45).raise_for_status()

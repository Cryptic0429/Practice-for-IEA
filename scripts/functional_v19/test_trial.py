"""Real HTTP inference parity and separately reported seen-condition decisions."""
import hashlib,json
import numpy as np
import requests
from functional_v1.common import ROOT,read,write,sha,stamp,EXP as BUDGET
from functional_v1.editor import load_rgb
from functional_v1.catalog import models
from .runtime import EXP,DATA
from .visual_review import EVAL,case_file
from .photometry import features

def main():
    receipt=EXP/'demo/http-integration.json';assert not receipt.exists()
    url='http://127.0.0.1:12346';catalog=requests.get(url+'/api/models',timeout=10).json()
    assert {m['id'] for m in catalog['models']}=={m['id'] for m in models()}
    assert catalog['default']=='functional-current-20261104'
    cases={split:{c['id']:c for c in read(case_file(split))} for split in ('native-development','old-native','decision-development')}
    training={c['id']:c for c in read(DATA/'cases.json')}
    progress=EXP/'demo/http-integration-progress.json';records=read(progress)['checks'] if progress.exists() else []
    done={(r['model_id'],r['split'],r['case_id'],r['language']) for r in records}
    for mid,meta in read(EXP/'development-selection.json')['models'].items():
        offline={split:{(r['case_id'],r['language']):r for r in map(json.loads,(EVAL/split/meta['evaluation_id']/'predictions.jsonl').read_text().splitlines())} for split in cases}
        tasks=[('native-development',cases['native-development'][cid],lang) for cid in ('native-dev-001','native-dev-006','native-dev-070') for lang in ('zh','en')]
        controls=list(cases['decision-development'].values())
        tasks += [('decision-development',next(c for c in controls if c['expected_action']==action),'zh') for action in ('keep','decline')]
        invalid=[(split,r) for split in ('native-development','old-native') for r in offline[split].values() if r['status']=='failed']
        if invalid:
            split,row=invalid[0];tasks.append((split,cases[split][row['case_id']],row['language']))
        tasks += [('seen-training-quality-condition',training[cid],'zh') for cid in ('visual-risk-01-severe-light','visual-risk-01-deep-light','visual-risk-01-mild-noise-light')]
        probe=read(EXP/'demo/case12-protocol.json')
        assert probe['model_id']==mid and probe['adapter_sha256']==meta['adapter_sha256'] and probe['used_for_checkpoint_selection'] is False
        assert sha(ROOT/probe['image'])==probe['input_sha256']
        historical={k:probe[k] for k in ('id','image','instructions')}
        tasks.extend(('historical-case12-not-independent-not-used-selection',historical,language) for language in ('zh','en'))
        for split,case,lang in tasks:
            if (mid,split,case['id'],lang) in done:continue
            with (ROOT/case['image']).open('rb') as handle:
                response=requests.post(url+'/api/edit',data=dict(instruction=case['instructions'][lang],model_id=mid),files={'image':('source.png',handle,'image/png')},timeout=240)
            actual=response.json();assert actual.get('request_id'),(response.status_code,actual)
            rid=actual['request_id'];path=BUDGET/'demo/requests'/rid/'request.json';saved=read(path)
            assert saved['training_use'] is False and saved['instruction']==case['instructions'][lang]
            saved['source']='agent_round19_HTTP_test_not_user_feedback';write(path,saved)
            assert np.array_equal(load_rgb(path.with_name('input.png')),load_rgb(ROOT/case['image']))
            assert np.array_equal(features(path.with_name('input.png')),features(ROOT/case['image']))
            expected=offline.get(split,{}).get((case['id'],lang))
            if expected:assert actual['status']==expected['status'],(actual,expected)
            identity=actual.get('identity');assert identity is not None,actual
            assert identity['adapter_sha256']==meta['adapter_sha256'] and identity['parent_adapter_sha256']==meta['parent_sha256']
            assert identity['runtime_family']=='functional_v19' and identity['training_route']=='photometric_bridge'
            assert identity['editor']==meta['editor'] and identity['training_data_sha256']==meta['dataset_sha256']
            assert identity['photometry_sha256']==sha(ROOT/'scripts/functional_v19/photometry.py')
            assert identity['bridge_source_sha256']==sha(ROOT/'scripts/functional_v19/bridge.py')
            if actual['status']=='executed':
                assert response.status_code==200
                output=requests.get(url+actual['after'],timeout=15);output.raise_for_status();digest=hashlib.sha256(output.content).hexdigest()
                if expected:assert digest==expected['output_sha256'] and actual['decision']==expected['decision']
                details=dict(output_sha256=digest,offline_byte_identical=expected is not None)
            elif actual['status']=='declined':
                assert response.status_code==200 and actual['after'] is None and actual['parameters'] is None
                assert not path.with_name('output.png').exists() and requests.get(url+f'/files/{rid}/output.png',timeout=10).status_code==404
                if expected:assert actual['decision']==expected['decision']
                details=dict(no_result_file=True,reason_code=actual['reason_code'])
            else:
                assert response.status_code==422 and actual['after'] is None and actual['parameters'] is None
                assert not path.with_name('output.png').exists() and requests.get(url+f'/files/{rid}/output.png',timeout=10).status_code==404
                assert expected and actual['detail']==expected['error']
                details=dict(no_result_file=True,offline_error_identical=True)
            seen=split=='seen-training-quality-condition'
            intended=case['decision'] if seen else None
            decision_correct=(actual.get('decision',{}).get('action')==intended['action'] and
                (intended['action']!='decline' or actual.get('reason_code')==intended['reason'])) if seen else None
            records.append(dict(model_id=mid,case_id=case['id'],split=split,language=lang,request_id=rid,status=actual['status'],
                input_RGB_and_features_exact=True,training_expected_decision=intended,observed_decision=actual.get('decision'),
                seen_condition_action_correct=decision_correct,visual_quality_newly_assessed=False,**details))
            write(progress,dict(at=stamp(),checks=records,independent_user_feedback=0))
        last=next(r for r in reversed(records) if r['model_id']==mid)
        response=requests.post(url+'/api/feedback',json=dict(request_id=last['request_id'],verdict='uncertain',comment='Agent R19 HTTP test only. Seen synthetic TRAIN condition; no independent user quality judgment.'),timeout=15);response.raise_for_status()
    requests.post('http://127.0.0.1:18282/unload',timeout=45).raise_for_status()
    assert requests.get(url+'/v19/evaluation-image/native-acceptance/photometric-bridge-e1/native-accept-001/zh/before',timeout=10).status_code==404
    write(receipt,dict(at=stamp(),status='passed_actual_HTTP_transport_identity_and_parity_checks',checks=records,
        models_registered=len(catalog['models']),default_unchanged=True,independent_user_feedback=0,training_use=False,
        seen_condition_action_checks=[r for r in records if r['split']=='seen-training-quality-condition'],
        quality_decision_failures_are_model_outcomes_not_hidden=True,new_visual_quality_assessments=0,
        worker_unloaded=True,no_external_rewriting=True,acceptance_closed=True))
    print('Actual HTTP requests',len(records),'; parity and model identity checked; seen-condition decisions separately recorded, not generalized.')

if __name__=='__main__':
    try:main()
    finally:requests.post('http://127.0.0.1:18282/unload',timeout=45).raise_for_status()

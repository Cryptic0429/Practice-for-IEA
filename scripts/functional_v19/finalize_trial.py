"""Bind observed browser calls, refusal boundary, feedback and occupied GPU time."""
import json
import requests
from functional_v1.common import ROOT,read,write,sha,stamp,GPUJob,EXP as BUDGET
EXP=ROOT/'experiments/functional-v19-20261107'

def main():
    target=EXP/'demo/browser-integration.json';assert not target.exists()
    selection=read(EXP/'development-selection.json');mid,meta=next(iter(selection['models'].items()))
    tasks=[('3b1ecd7a980f4bd29fbeca21262ee666',mid,'historical-case12-brown'),
           ('0ab973af88ed4dc4a35f452f2fd0e0ef','functional-v15-shadow-correction-20261127','historical-case12-brown'),
           ('4d8c6f7bd62c41729616b2564d4d8b4a',mid,'native-dev-070')]
    http=read(EXP/'demo/http-integration.json');records=[]
    for rid,model,cid in tasks:
        path=BUDGET/'demo/requests'/rid/'request.json';r=read(path)
        assert r['model_id']==model and r['source']=='agent_round19_browser_test_not_user_feedback' and not r['training_use']
        actual_meta=meta if model==mid else read(ROOT/'experiments/functional-v15-20261106/development-selection.json')['models'][model]
        assert r['identity']['adapter_sha256']==actual_meta['adapter_sha256'] and r['identity']['editor']==actual_meta['editor']
        assert r['identity']['parent_adapter_sha256']==actual_meta['parent_sha256']
        if cid=='historical-case12-brown':
            assert r['instruction']=='Please, add brown tone' and r['status']=='executed'
            prior=http['checks'] if model==mid else read(ROOT/'experiments/functional-v15-20261106/demo/http-integration.json')['checks']
            expected=next(x for x in prior if x['case_id']==cid and x['language']=='en')
            assert sha(path.with_name('output.png'))==expected['output_sha256']
            if model==mid:assert read(EXP/'demo/local-download-check.json')['sha256']==expected['output_sha256']
            details=dict(output_sha256=expected['output_sha256'],historical_case12_not_independent=True)
        else:
            assert r['instruction']=='increase the brightness a lot' and r['status']=='declined' and r['reason_code']=='unreliable_quality'
            assert not path.with_name('output.png').exists()
            assert requests.get(f'http://127.0.0.1:12346/files/{rid}/output.png',timeout=10).status_code==404
            expected=next(x for x in http['checks'] if x['case_id']==cid and x['language']=='en')
            assert r['decision']==expected['observed_decision']
            details=dict(no_result_image=True,image_route_status=404,refusal_correctness_not_inferred_from_completion_count=True)
        records.append(dict(request_id=rid,model_id=model,case_id=cid,language='en',status=r['status'],**details))
    all_feedback=list(map(json.loads,(BUDGET/'demo/feedback.jsonl').read_text().splitlines()))
    agent=[r for r in all_feedback if r.get('source','').startswith('agent_round19_')]
    assert len(agent)==2 and all(not r['training_use'] for r in agent)
    independent=[]
    for r in all_feedback:
        request=read(BUDGET/'demo/requests'/r['request_id']/'request.json')
        if request['model_id'] in selection['models'] and not r.get('source','').startswith('agent_'):independent.append(r)
    write(EXP/'demo/agent-feedback-records.json',dict(records=agent,independent_user_feedback=len(independent),training_use=False,
        note='The HTTP generic feedback comment says synthetic TRAIN, but its bound request is historical case12. This is an agent comment scope error, not a user assessment or training label; raw comment retained.'))
    write(EXP/'demo/independent-feedback-audit.json',dict(at=stamp(),registered_model_ids=list(selection['models']),count=len(independent),records=independent,
        method='Read actual feedback and resolve its request model; agent-prefixed test records excluded. Uploads/feedback remain training_use=false.'))
    requests.post('http://127.0.0.1:18282/unload',timeout=45).raise_for_status()
    health=requests.get('http://127.0.0.1:18282/health',timeout=10).json();assert health['loaded_model'] is None
    catalog=requests.get('http://127.0.0.1:12346/api/models',timeout=10).json();assert len(catalog['models'])==28 and catalog['default']=='functional-current-20261104'
    ledger=read(BUDGET/'resources.json');total=GPUJob.consumed(ledger)
    start=next(j['started_epoch'] for j in ledger['jobs'] if j['id']=='694548a1a09647fb9eb349aaa525d745')
    jobs=[j for j in ledger['jobs'] if j['started_epoch']>=start]
    assert all(j.get('ended_epoch') for j in jobs)
    assert all(str(j['gpu']) in ('0','2','3') for j in jobs)
    resource=dict(at=stamp(),occupied_GPU_hours=total,cap=24,remaining_GPU_hours=24-total,
        round19_window_GPU_hours=sum((j['ended_epoch']-j['started_epoch'])/3600 for j in jobs),jobs=jobs,worker=health,
        original_GPU1_service_preserved=True,scope='All ledger occupancy after first R19 job through this trial audit, including R15 parent comparison; future work excluded.')
    write(EXP/'resource-summary.json',resource)
    screenshots={str(p.relative_to(ROOT)):sha(p) for p in (EXP/'demo').glob('browser-*.jpg')}
    assert all(str((EXP/'demo'/name).relative_to(ROOT)) in screenshots for name in ('browser-case12-result.jpg','browser-parent-case12.jpg','browser-no-result-refusal.jpg'))
    write(target,dict(at=stamp(),status='actual_browser_upload_edit_parent_switch_download_refusal_feedback_verified',requests=records,
        model_switch_cleared_old_result=True,alternative_model_actually_inferred=True,screenshots=screenshots,
        independent_user_feedback=len(independent),agent_feedback_records=2,worker_unloaded=True,default_unchanged=True,
        occupied_GPU_hours=total,no_external_rewriting=True,
        limitations=['Known historical/dev inputs; agent UI checks are not independent user approval.',
                     'Chinese star-field result has major noise; English refusal is separate from requested-edit completion.',
                     'Both checkpoints failed; no controlled evaluation, replication or final acceptance.']))
    print('Verified3 actual browser calls across2models,download/no-result refusal/feedback. GPUh',total,'/24; userfeedback',len(independent))

if __name__=='__main__':main()

"""Bind actual browser/API evidence, keeping all feedback explicitly agent-only."""
import datetime
import json
import requests
import numpy as np
from functional_v1.common import ROOT,read,write,sha,stamp,GPUJob,EXP as BUDGET_EXP
from functional_v1.editor import load_rgb
from .runtime import EXP


def main():
    target=EXP/'demo/browser-integration.json';assert not target.exists()
    selection=read(EXP/'development-selection.json');model=next(iter(selection['models'].values()))
    base=EXP/'evaluation'/model['editor'];records=[]
    tasks=[('e058d386a96341b3ac1b4acfb0e052fc',model['model_id'],'native-dev-001'),
           ('4f2b354aeccd46beb0c5da00d1de3e19','functional-v14-visual-late4-20261127','native-dev-001'),
           ('f6ae00ef73694916aa3b6add92889cc1',model['model_id'],'visual-risk-01-severe-light')]
    for rid,mid,cid in tasks:
        path=BUDGET_EXP/'demo/requests'/rid/'request.json';request=read(path)
        assert request['model_id']==mid and not request['training_use']
        assert request['source']=='agent_round15_browser_test_not_user_feedback'
        if mid==model['model_id']:meta=model;folder=base/'native-development'/meta['evaluation_id']
        else:
            prior=read(ROOT/'experiments/functional-v14-20261106/development-selection.json')
            meta=prior['models'][mid]
            folder=ROOT/'experiments/functional-v14-20261106/evaluation'/meta['editor']/'native-development'/meta['evaluation_id']
        for identity,key in [('adapter_sha256','adapter_sha256'),('editor','editor'),('schema','schema'),('parent_adapter_sha256','parent_sha256')]:
            assert request['identity'][identity]==meta[key]
        if cid=='native-dev-001':
            expected=next(r for r in map(json.loads,(folder/'predictions.jsonl').read_text().splitlines()) if r['case_id']==cid and r['language']=='zh')
            assert request['status']=='executed' and request['decision']==expected['decision']
            assert sha(path.with_name('output.png'))==expected['output_sha256']
            assert np.array_equal(load_rgb(path.with_name('input.png')),load_rgb(ROOT/expected['input_image']))
            if mid==model['model_id']:assert read(EXP/'demo/local-download-check.json')['sha256']==expected['output_sha256']
            records.append(dict(request_id=rid,model_id=mid,case_id=cid,status='executed',offline_byte_identical=True,
                output_sha256=expected['output_sha256'],actual_model_switch_inference=mid!=model['model_id']))
        else:
            assert request['status']=='declined' and request['reason_code']=='unreliable_quality'
            assert not path.with_name('output.png').exists()
            assert requests.get(f'http://127.0.0.1:12346/files/{rid}/output.png',timeout=15).status_code==404
            records.append(dict(request_id=rid,model_id=mid,case_id=cid,status='declined',no_result_image=True,
                image_route_status=404,seen_synthetic_training_condition_not_generalization=True))
    feedback=[r for r in map(json.loads,(BUDGET_EXP/'demo/feedback.jsonl').read_text().splitlines()) if r.get('source','').startswith('agent_round15_')]
    assert len(feedback)==2 and all(not r['training_use'] for r in feedback)
    assert sum(r['source']=='agent_round15_browser_test_not_user_feedback' for r in feedback)==1
    write(EXP/'demo/agent-feedback-records.json',dict(records=feedback,independent_user_feedback=0,training_use=False))
    shots={str(p.relative_to(ROOT)):sha(p) for p in (EXP/'demo').glob('browser-*.jpg')};assert len(shots)==3
    requests.post('http://127.0.0.1:18282/unload',timeout=45).raise_for_status()
    health=requests.get('http://127.0.0.1:18282/health',timeout=10).json();assert health['loaded_model'] is None
    catalog=requests.get('http://127.0.0.1:12346/api/models',timeout=10).json()
    assert len(catalog['models'])==25 and catalog['default']=='functional-current-20261104'
    ledger=read(BUDGET_EXP/'resources.json');total=GPUJob.consumed(ledger);assert ledger['limit_gpu_hours']==24
    freeze=read(EXP/'training-protocol.json')
    start=datetime.datetime.fromisoformat(freeze['at']).timestamp()
    jobs=[j for j in ledger['jobs'] if j['started_epoch']>=start]
    assert all('ended_epoch' in j for j in jobs)
    r17=[j for j in jobs if j['stage']=='v17-TRAIN-resolution-diagnostic']
    r15=[j for j in jobs if j not in r17]
    assert all(j['stage'].startswith(('v15-','interactive-trial-functional-v15-','interactive-trial-functional-v14-')) for j in r15)
    hours=lambda values:sum((j['ended_epoch']-j['started_epoch'])/3600 for j in values)
    resource=dict(at=stamp(),occupied_GPU_hours=total,cap=24,remaining_GPU_hours=24-total,
        round15_GPU_hours=hours(r15),round17_diagnostic_GPU_hours=hours(r17),jobs=jobs,
        original_GPU1_service_preserved=True,worker=health)
    write(EXP/'demo/resource-after-browser.json',resource);write(EXP/'resource-summary.json',resource)
    write(target,dict(at=stamp(),status='actual_browser_upload_edit_model_switch_download_feedback_and_records_verified',
        requests=records,model_switch_cleared_old_result=True,alternative_model_actually_inferred=True,
        screenshots=shots,feedback_records=2,independent_user_feedback=0,worker_unloaded=True,
        records_page='http://127.0.0.1:12346/v15',models_registered=25,default_unchanged=True,
        occupied_GPU_hours=total,actual_external_rewriting=False,
        limitations=['Quality-refusal probe is a seen synthetic training condition, not an unknown real-camera photograph.',
                    'All feedback is agent testing; no independent user usability evidence.',
                    'All development sets are adaptive/reused; final acceptance stays closed.']))
    print('Verified3 actual browser requests across2 real models, download, no-image refusal,2 agent feedback records. GPUh',total,'/24; R15',hours(r15),'R17',hours(r17))


if __name__=='__main__':main()

"""Verify browser requests, preserving agent feedback as distinct from user evidence."""
import json
import requests
import numpy as np
from functional_v1.common import ROOT,read,write,sha,stamp,GPUJob,EXP as BUDGET_EXP
from functional_v1.editor import load_rgb
from .web_records import EXP,EVAL


def main():
    target=EXP/'demo/browser-integration.json';assert not target.exists()
    selection=read(EXP/'development-selection.json');records=[]
    request_root=BUDGET_EXP/'demo/requests'
    tasks=[('5155ab9b36a74964bcc0ac28453db099','functional-v14-visual-late4-20261127','native-dev-001'),
           ('3ec9728cdb0f4dfda4cba849e400f14c','functional-v14-visual-late4-20261127','visual-risk-01-severe-light')]
    for rid,mid,cid in tasks:
        meta=selection['models'][mid];p=request_root/rid/'request.json';r=read(p)
        assert r['model_id']==mid and r['training_use'] is False
        assert r['source']=='agent_round14_browser_test_not_user_feedback'
        for a,b in [('adapter_sha256','adapter_sha256'),('editor','editor'),('schema','schema'),('parent_adapter_sha256','parent_sha256')]:
            assert r['identity'][a]==meta[b]
        if cid=='native-dev-001':
            expected=next(x for x in map(json.loads,(EVAL/'native-development'/meta['evaluation_id']/'predictions.jsonl').read_text().splitlines()) if x['case_id']==cid and x['language']=='zh')
            assert r['status']=='executed' and r['decision']==expected['decision']
            assert sha(p.with_name('output.png'))==expected['output_sha256']=='018d4db48fd1f5b402edbe7c58600d12ed7b164923f9fb3b3ae33c11b8984765'
            assert np.array_equal(load_rgb(p.with_name('input.png')),load_rgb(ROOT/expected['input_image']))
            receipt=read(EXP/'demo/local-download-check.json')
            assert receipt['sha256']==expected['output_sha256']
            records.append(dict(request_id=rid,model_id=mid,case_id=cid,status='executed',offline_byte_identical=True,download=receipt))
        else:
            assert r['status']=='declined' and r['reason_code']=='unreliable_quality' and not p.with_name('output.png').exists()
            assert requests.get(f'http://127.0.0.1:12346/files/{rid}/output.png',timeout=10).status_code==404
            records.append(dict(request_id=rid,model_id=mid,case_id=cid,status='declined',no_result_image=True,image_route_status=404,training_condition_probe_not_independent_generalization=True))
    feedback=[r for r in map(json.loads,(BUDGET_EXP/'demo/feedback.jsonl').read_text().splitlines()) if r.get('source','').startswith('agent_round14_')]
    assert len(feedback)==2 and all(r['training_use'] is False for r in feedback)
    assert sum(r['source']=='agent_round14_browser_test_not_user_feedback' for r in feedback)==1
    write(EXP/'demo/agent-feedback-records.json',dict(records=feedback,independent_user_feedback=False,automatically_used_for_training=False))
    shots={str(p.relative_to(ROOT)):sha(p) for p in (EXP/'demo').glob('browser-*.png')};assert len(shots)==3
    health=requests.get('http://127.0.0.1:18282/health',timeout=10).json();assert health['loaded_model'] is None
    ledger=read(BUDGET_EXP/'resources.json');consumed=GPUJob.consumed(ledger)
    assert ledger['limit_gpu_hours']==24
    write(EXP/'demo/resource-after-browser.json',dict(at=stamp(),occupied_GPU_hours=consumed,cap=24,worker=health))
    write(target,dict(at=stamp(),status='actual_browser_upload_edit_switch_download_feedback_and_records_verified',requests=records,
        model_switch_cleared_old_result=True,alternative_model_not_inferred_in_this_browser_phase=True,screenshots=shots,feedback_records=2,independent_user_feedback=0,worker_unloaded=True,
        records_page='http://127.0.0.1:12346/v14',models_registered=24,default_unchanged=True,
        occupied_GPU_hours=consumed,actual_external_rewriting=False,
        limitations=['Quality refusal uses a seen synthetic training condition, not an unknown real-camera photograph.',
                    'All feedback is agent integration testing; no independent user usability result.',
                    'Native development and controlled development have been repeatedly used; acceptance remains closed.']))
    print('Browser parity, downloaded bytes, refusal without image, 2 agent feedback records verified; GPUh',consumed)


if __name__=='__main__':main()

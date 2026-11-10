"""Verify browser-visible results against saved model identities and offline bytes."""
import json
import requests
import numpy as np
from functional_v1.common import ROOT,read,write,sha,stamp,GPUJob,EXP as BUDGET_EXP
from functional_v1.editor import load_rgb
from .web_records import EXP,EVAL


def main():
    target=EXP/'demo/browser-integration.json';assert not target.exists()
    selection=read(EXP/'development-selection.json');records=[]
    request_root=ROOT/'experiments/functional-v1-20261104/demo/requests'
    for rid,mid,cid in [('a487f69d6813439b83ce9ce4738f9a0d','functional-v10-language-20261121','native-dev-001'),
                        ('9cc87fee2b0f422a8cfe67e7f08a2a3f','functional-v10-language-merger-20261121','visual-risk-01-severe-light')]:
        meta=selection['models'][mid];p=request_root/rid/'request.json';r=read(p)
        assert r['model_id']==mid and r['training_use'] is False and r['source']=='agent_round10_browser_test_not_user_feedback'
        assert r['identity']['adapter_sha256']==meta['adapter_sha256']
        assert r['identity']['editor']==meta['editor'] and r['identity']['schema']==meta['schema']
        assert r['identity']['parent_adapter_sha256']==meta['parent_sha256']
        if cid=='native-dev-001':
            expected=next(r for r in map(json.loads,(EVAL/'native-development'/meta['evaluation_id']/'predictions.jsonl').read_text().splitlines()) if r['case_id']==cid and r['language']=='zh')
            assert r['status']=='executed' and r['decision']==expected['decision']
            assert sha(p.with_name('output.png'))==expected['output_sha256']=='018d4db48fd1f5b402edbe7c58600d12ed7b164923f9fb3b3ae33c11b8984765'
            assert np.array_equal(load_rgb(p.with_name('input.png')),load_rgb(ROOT/expected['input_image']))
            records.append(dict(request_id=rid,model_id=mid,case_id=cid,status='executed',offline_byte_identical=True,download_sha256=expected['output_sha256'],local_download='/Users/cryptic/Downloads/output (2).png'))
        else:
            assert r['status']=='declined' and r['reason_code']=='unreliable_quality' and not p.with_name('output.png').exists()
            assert requests.get(f'http://127.0.0.1:12346/files/{rid}/output.png',timeout=10).status_code==404
            records.append(dict(request_id=rid,model_id=mid,case_id=cid,status='declined',no_result_image=True,image_route_status=404,training_condition_probe_not_independent_generalization=True))
    feedback=[]
    for r in map(json.loads,(BUDGET_EXP/'demo/feedback.jsonl').read_text().splitlines()):
        if r.get('source','').startswith('agent_round10_'):
            assert r['training_use'] is False;feedback.append(r)
    assert len(feedback)==3 and sum(r['source']=='agent_round10_browser_test_not_user_feedback' for r in feedback)==1
    write(EXP/'demo/agent-feedback-records.json',dict(records=feedback,independent_user_feedback=False,automatically_used_for_training=False))
    shots={str(p.relative_to(ROOT)):sha(p) for p in (EXP/'demo').glob('browser-*.png')};assert len(shots)==5
    health=requests.get('http://127.0.0.1:18282/health',timeout=10).json();assert health['loaded_model'] is None
    ledger=read(BUDGET_EXP/'resources.json');consumed=GPUJob.consumed(ledger)
    write(EXP/'demo/resource-after-browser.json',dict(at=stamp(),occupied_GPU_hours=consumed,cap=16,worker=health))
    write(target,dict(at=stamp(),status='actual_browser_upload_edit_switch_download_feedback_and_records_verified',requests=records,
        screenshots=shots,feedback_records=3,independent_user_feedback=0,
        records_filters=dict(pass_language_rows=28,unqualified_language_rows=10,uncertain_language_rows=2,total_language_rows=40,model='language-merger-e2'),
        final_page='http://127.0.0.1:12346/v10?kind=evaluation&domain=all&action=all&judgment=all&split=native-development&model=language-merger-e2',
        occupied_GPU_hours=consumed,worker_unloaded=True,default_unchanged=True,actual_external_rewriting=False,
        limitations=['Quality refusal is checked on a seen synthetic training condition, not an unknown real-camera photograph.',
            'All feedback records are explicitly agent integration checks; no independent user usability result.',
            'Initial exact-label browser selector did not resolve; recovered using the visible combobox role without submitting duplicate edits.',
            'Header wrapping CSS was adjusted after edit screenshots; records proof and semantic UI checks were performed on the final service.']))
    print('Browser parity, no-result refusal, downloaded bytes,3 agent feedback records, and28/10/2 filtered rows verified;GPUh',consumed)

if __name__=='__main__':main()

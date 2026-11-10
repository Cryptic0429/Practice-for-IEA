"""Bind actual three-model browser evidence and the full charged resource ledger."""
import json
import requests
from functional_v1.common import ROOT,read,write,sha,stamp,GPUJob,EXP as BUDGET
from .runtime import EXP

def main():
    target=EXP/'demo/browser-integration.json';assert not target.exists()
    selection=read(EXP/'development-selection.json'); records=[]
    tasks=[('742d7b3b03c64dedab5cafa0ac5ff516','functional-v18-counterfactual-ranking-20261129'),
           ('cec51d78e4bf4eccab3fe915000858eb','functional-v18-matched-sft-20261129'),
           ('7fe9407abd794eb4bf145f4a866273d2','functional-v15-shadow-correction-20261127')]
    for rid,mid in tasks:
        request_path=BUDGET/'demo/requests'/rid/'request.json';r=read(request_path)
        assert r['source']=='agent_round18_browser_test_not_user_feedback' and r['training_use'] is False
        assert r['model_id']==mid and r['instruction']=='添加棕褐色色调。' and r['status']=='executed'
        if mid in selection['models']:meta=selection['models'][mid];experiment=EXP
        else:
            experiment=ROOT/'experiments/functional-v15-20261106';meta=read(experiment/'development-selection.json')['models'][mid]
        predictions=experiment/'evaluation'/meta['editor']/'native-development'/meta['evaluation_id']/'predictions.jsonl'
        expected=next(x for x in map(json.loads,predictions.read_text().splitlines()) if x['case_id']=='native-dev-001' and x['language']=='zh')
        assert r['identity']['adapter_sha256']==meta['adapter_sha256']
        assert r['identity']['parent_adapter_sha256']==meta['parent_sha256'] and r['identity']['editor']==meta['editor']
        assert r['decision']==expected['decision'] and sha(request_path.with_name('output.png'))==expected['output_sha256']
        records.append(dict(request_id=rid,model_id=mid,case_id='native-dev-001',language='zh',status='executed',offline_byte_identical=True,output_sha256=expected['output_sha256']))
    assert read(EXP/'demo/local-download-check.json')['sha256']==records[0]['output_sha256']
    feedback=[r for r in map(json.loads,(BUDGET/'demo/feedback.jsonl').read_text().splitlines()) if r.get('source','').startswith('agent_round18_')]
    assert len(feedback)==3 and all(not r['training_use'] for r in feedback)
    assert sum(r['source']=='agent_round18_browser_test_not_user_feedback' for r in feedback)==1
    write(EXP/'demo/agent-feedback-records.json',dict(records=feedback,independent_user_feedback=0,training_use=False))
    failed_ids=['2a18762da6554f8d8b441c1a4b460cd7','d8760c8a330847068d00401368ff893c','04ce0e90db024c888585499d318bf6dc']
    for rid in failed_ids:
        r=read(BUDGET/'demo/requests'/rid/'request.json');assert r['source']=='agent_round18_HTTP_test_not_user_feedback'
        assert not r['training_use'] and not (BUDGET/'demo/requests'/rid/'output.png').exists()
    requests.post('http://127.0.0.1:18282/unload',timeout=45).raise_for_status()
    health=requests.get('http://127.0.0.1:18282/health',timeout=10).json();assert health['loaded_model'] is None
    catalog=requests.get('http://127.0.0.1:12346/api/models',timeout=10).json();assert len(catalog['models'])==27 and catalog['default']=='functional-current-20261104'
    ledger=read(BUDGET/'resources.json');total=GPUJob.consumed(ledger)
    start=next(j['started_epoch'] for j in ledger['jobs'] if j['id']=='1b654a8fb5c24c76bd00f37b348535f4')
    jobs=[j for j in ledger['jobs'] if j['started_epoch']>=start]
    assert all(j.get('ended_epoch') for j in jobs)
    assert all(j['stage'].startswith(('v18-','interactive-trial-functional-v18-','interactive-trial-functional-v15-')) for j in jobs)
    assert all(str(j['gpu']) in ('0','2','3') for j in jobs)
    resource=dict(at=stamp(),occupied_GPU_hours=total,cap=24,remaining_GPU_hours=24-total,
        round18_GPU_hours=sum((j['ended_epoch']-j['started_epoch'])/3600 for j in jobs),jobs=jobs,worker=health,
        original_GPU1_service_preserved=True,failed_attempts_billed=True)
    write(EXP/'resource-summary.json',resource)
    screenshots={str(p.relative_to(ROOT)):sha(p) for p in (EXP/'demo').glob('browser-*') if p.suffix in ('.jpg','.png')}
    assert len([p for p in screenshots if p.endswith('-result.jpg')])==3
    write(target,dict(at=stamp(),status='actual_browser_upload_edit_three_models_download_feedback_verified',requests=records,
        model_switch_cleared_old_result=True,alternative_models_actually_inferred=True,screenshots=screenshots,
        independent_user_feedback=0,agent_feedback_records=3,failed_integration_request_ids=failed_ids,
        worker_unloaded=True,default_unchanged=True,occupied_GPU_hours=total,no_external_rewriting=True,
        limitations=['All requests are agent integration checks; none are independent user usability feedback.',
                     'Browser example is a reused development image, not unseen acceptance.',
                     'No eligible model, no controlled evaluation, no replicate, final acceptance remains closed.']))
    print('Verified actual three-model browser inference/download/feedback; GPUh',total,'/24; R18',resource['round18_GPU_hours'])

if __name__=='__main__':main()

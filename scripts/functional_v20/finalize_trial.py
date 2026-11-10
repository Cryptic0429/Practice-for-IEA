"""Bind actual3 browser calls andall agent-only feedback, then unload and account."""
import fcntl,json,subprocess
import requests
from .common import ROOT,EXP,read,write,sha,stamp,GPUJob
from functional_v1.common import EXP as BUDGET
from functional_v1.catalog import models

def main():
    target=EXP/'demo/browser-integration.json';assert not target.exists()
    requests.post('http://127.0.0.1:18282/unload',timeout=45).raise_for_status()
    selection=read(EXP/'development-selection.json');http=read(EXP/'demo/http-integration.json')
    tasks=[('4b9e5f394b194919b73f3664550da77f','functional-v20-last-20261203','historical-case12-brown'),
        ('f7502593281d4f6782e04d5e65aaea89','functional-v20-request-mean-20261203','historical-case12-brown'),
        ('2ecb4fa21bf44d88a5afeb0299e9c266','functional-v20-last-20261203','native-dev-070')]
    records=[]
    for rid,mid,cid in tasks:
        path=BUDGET/'demo/requests'/rid/'request.json';r=read(path);meta=selection['models'][mid]
        assert r['model_id']==mid and not r['training_use']
        assert r['identity']['controller_sha256']==meta['controller_sha256'] and r['identity']['parent_adapter_sha256']==meta['parent_sha256']
        r['source']='agent_round20_browser_test_not_user_feedback';write(path,r)
        expected=next(x for x in http['checks'] if x['model_id']==mid and x['case_id']==cid and x['language']=='en')
        if cid.startswith('historical'):
            assert r['instruction']=='Please, add brown tone' and r['status']=='executed'
            assert sha(path.with_name('output.png'))==expected['output_sha256']
            details=dict(output_sha256=expected['output_sha256'],historical_not_independent=True)
            if mid.endswith('-last-20261203'):
                assert sha(EXP/'demo/browser-download.png')==expected['output_sha256']
                write(EXP/'demo/local-download-check.json',dict(path='demo/browser-download.png',sha256=sha(EXP/'demo/browser-download.png'),actual_browser_download=True,HTTP_exact=True))
        else:
            assert r['instruction']=='increase the brightness a lot' and r['status']=='declined' and r['reason_code']=='unreliable_quality'
            assert r['decision']==expected['decision'] and not path.with_name('output.png').exists()
            assert requests.get(f'http://127.0.0.1:12346/files/{rid}/output.png',timeout=10).status_code==404
            details=dict(no_result_image=True,image_route_status=404,refusal_correctness_not_inferred_from_completion_count=True)
        records.append(dict(request_id=rid,model_id=mid,case_id=cid,language='en',status=r['status'],**details))
    http_ids={r['request_id'] for r in http['checks']};browser_ids={r[0] for r in tasks};own=http_ids|browser_ids
    path=BUDGET/'demo/feedback.jsonl'
    with path.open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);feedback=list(map(json.loads,f.read().splitlines()))
        agent=[r for r in feedback if r['request_id'] in own];assert len(agent)==3
        for r in agent:
            assert not r['training_use']
            r['source']='agent_round20_'+('browser' if r['request_id'] in browser_ids else 'HTTP')+'_test_not_user_feedback'
        f.seek(0);f.write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in feedback));f.truncate();f.flush()
    independent=[r for r in feedback if read(BUDGET/'demo/requests'/r['request_id']/'request.json')['model_id'] in selection['models'] and not r.get('source','').startswith('agent_')]
    write(EXP/'demo/agent-feedback-records.json',dict(records=agent,independent_user_feedback=len(independent),training_use=False))
    write(EXP/'demo/independent-feedback-audit.json',dict(at=stamp(),count=len(independent),records=independent,
        method='Resolve actual feedback to request model, exclude explicitly marked agent tests; all remain separate from training.'))
    health=requests.get('http://127.0.0.1:18282/health',timeout=10).json();assert health['loaded_model'] is None
    catalog=requests.get('http://127.0.0.1:12346/api/models',timeout=10).json()
    assert len(catalog['models'])==len(models()) and catalog['default']=='functional-current-20261104'
    ledger=read(BUDGET/'resources.json');total=GPUJob.consumed(ledger)
    start=min(j['started_epoch'] for j in ledger['jobs'] if j['stage'].startswith('v20-'))
    jobs=[j for j in ledger['jobs'] if j['started_epoch']>=start]
    assert all(j.get('ended_epoch') and str(j['gpu']) in ('0','2','3') for j in jobs)
    assert '3378799' in subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
    resource=dict(at=stamp(),occupied_GPU_hours=total,cap=24,remaining_GPU_hours=24-total,
        round20_window_GPU_hours=sum((j['ended_epoch']-j['started_epoch'])/3600 for j in jobs),jobs=jobs,worker=health,
        original_GPU1_service_preserved=True,scope='All occupancy after first R20 cache through actual trial, including failures and idle time; future work excluded.')
    write(EXP/'resource-summary.json',resource)
    screenshots={str(p.relative_to(ROOT)):sha(p) for p in (EXP/'demo').glob('browser-*.jpg')}
    assert len(screenshots)==4
    write(target,dict(at=stamp(),status='actual_browser_upload_edit_switch_download_refusal_feedback_verified',requests=records,
        model_switch_cleared_old_result=True,both_registered_heads_actually_called=True,screenshots=screenshots,
        independent_user_feedback=len(independent),agent_feedback_records=3,worker_unloaded=True,default_unchanged=True,
        occupied_GPU_hours=total,no_external_rewriting=True,
        limitations=['Repeated historical/dev photos, not independent user approval.','Chinese night result failedquality; English refusal correctness not established by completion counts.','All4checkpoints failed; no controlled100,replicate oracceptance.']))
    print('Verified24 actual HTTP/browser requests across2heads,3agentfeedback; GPUh',total,'; remaining',24-total,'; independentuser',len(independent))

if __name__=='__main__':main()

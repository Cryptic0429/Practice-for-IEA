"""Bind four actual browser calls, agent-only feedback and occupied GPU time."""
import fcntl,json,subprocess
import requests
from .common import ROOT,EXP,read,write,sha,stamp,GPUJob
from functional_v1.common import EXP as BUDGET
from functional_v1.catalog import models


def main():
    target=EXP/'demo/browser-integration.json';assert not target.exists()
    requests.post('http://127.0.0.1:18282/unload',timeout=45).raise_for_status()
    selection=read(EXP/'development-selection.json');http=read(EXP/'demo/http-integration.json')
    tasks=[('b9a4490ba1fa4bab96d55b767e9b52f5','functional-v21-quality-20261205','historical-case12-brown','en'),
        ('0ca5ad6bde6f4443a7bcc688e7704d59','functional-v21-magnitude-20261205','historical-case12-brown','en'),
        ('c23ceecd44c84e48ac7ca6481666b711','functional-v21-quality-20261206','native-dev-070','zh'),
        ('d6ba01a1b14b4b38a65d5f63be224b9e','functional-v21-quality-20261205','native-dev-070','zh')]
    records=[]
    for rid,mid,cid,language in tasks:
        path=BUDGET/'demo/requests'/rid/'request.json';r=read(path);meta=selection['models'][mid]
        assert r['model_id']==mid and not r['training_use']
        assert r['identity']['controller_sha256']==meta['controller_sha256'] and r['identity']['parent_adapter_sha256']==meta['parent_sha256']
        assert r['identity']['shared_weight_policy']==meta['policy']
        r['source']='agent_round21_browser_test_not_user_feedback';write(path,r)
        expected=next(x for x in http['checks'] if x['model_id']==mid and x['case_id']==cid and x['language']==language)
        assert r['status']==expected['status'] and r['decision']==expected['decision']
        if r['status']=='executed':
            assert sha(path.with_name('output.png'))==expected['output_sha256']
            details=dict(output_sha256=expected['output_sha256'],HTTP_exact=True)
            if cid.startswith('historical') and mid=='functional-v21-quality-20261205':
                assert sha(EXP/'demo/browser-download.png')==expected['output_sha256']
                write(EXP/'demo/local-download-check.json',dict(path='demo/browser-download.png',sha256=expected['output_sha256'],actual_browser_download=True,HTTP_exact=True))
        else:
            assert r['reason_code']=='unreliable_quality' and not path.with_name('output.png').exists()
            assert requests.get(f'http://127.0.0.1:12346/files/{rid}/output.png',timeout=10).status_code==404
            details=dict(no_result_image=True,image_route_status=404)
        assert r['instruction']==('Please, add brown tone' if cid.startswith('historical') else '大幅提高亮度。')
        records.append(dict(request_id=rid,model_id=mid,case_id=cid,language=language,status=r['status'],
            historical_not_independent=cid.startswith('historical'),visual_quality_newly_scored=False,**details))
    own={r['request_id'] for r in http['checks']+records};browser_ids={r[0] for r in tasks};assert len(own)==31
    path=BUDGET/'demo/feedback.jsonl'
    with path.open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);feedback=list(map(json.loads,f.read().splitlines()))
        agent=[r for r in feedback if r['request_id'] in own];assert len(agent)==4
        for r in agent:
            assert not r['training_use']
            r['source']='agent_round21_'+('browser' if r['request_id'] in browser_ids else 'HTTP')+'_test_not_user_feedback'
        f.seek(0);f.write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in feedback));f.truncate();f.flush()
    independent=[r for r in feedback if read(BUDGET/'demo/requests'/r['request_id']/'request.json')['model_id'] in selection['models'] and not r.get('source','').startswith('agent_')]
    write(EXP/'demo/agent-feedback-records.json',dict(records=agent,independent_user_feedback=len(independent),training_use=False))
    write(EXP/'demo/independent-feedback-audit.json',dict(at=stamp(),count=len(independent),records=independent,
        method='Resolve actual feedback to request model, exclude explicitly marked agent tests; never auto-admit uploads or feedback to training.'))
    health=requests.get('http://127.0.0.1:18282/health',timeout=10).json();assert health['loaded_model'] is None
    catalog=requests.get('http://127.0.0.1:12346/api/models',timeout=10).json()
    assert len(catalog['models'])==len(models())==33 and catalog['default']=='functional-current-20261104'
    ledger=read(BUDGET/'resources.json');total=GPUJob.consumed(ledger)
    start=min(j['started_epoch'] for j in ledger['jobs'] if j['stage'].startswith('v21-'))
    jobs=[j for j in ledger['jobs'] if j['started_epoch']>=start]
    assert all(j.get('ended_epoch') and str(j['gpu']) in ('0','2','3') for j in jobs)
    assert '3378799' in subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
    resource=dict(at=stamp(),occupied_GPU_hours=total,cap=24,remaining_GPU_hours=24-total,
        round21_window_GPU_hours=sum((j['ended_epoch']-j['started_epoch'])/3600 for j in jobs),jobs=jobs,worker=health,
        original_GPU1_service_preserved=True,scope='All occupancy from first R21 controlled input cache through actual trial, including load/failure/idle time; earlier CPU-only work uses no GPU; future work excluded.')
    write(EXP/'resource-summary.json',resource)
    screenshots={str(p.relative_to(ROOT)):sha(p) for p in (EXP/'demo').glob('browser-*.jpg')};assert len(screenshots)==5
    write(target,dict(at=stamp(),status='actual_browser_upload_edit_switch_download_refusal_feedback_verified',requests=records,
        model_switch_cleared_old_result=True,all_three_registered_variants_actually_called=True,two_independent_heads=True,screenshots=screenshots,
        independent_user_feedback=len(independent),agent_feedback_records=4,worker_unloaded=True,default_unchanged=True,
        occupied_GPU_hours=total,no_external_rewriting=True,
        limitations=['Historical/development photos, not independent user approval.','Seed2 night result is a confirmed major-quality failure; seed1 refusal is not an edit completion.',
            'Seed1 magnitude/quality share exact weights; policies are not independent training runs.','Locked seed replication failed; acceptance remains sealed.']))
    print('Verified31 actual HTTP/browser calls,4agentfeedback; GPUh',total,'; remaining',24-total,'; independentuser',len(independent))


if __name__=='__main__':main()

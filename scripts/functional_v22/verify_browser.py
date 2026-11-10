"""Bind actual browser test files and mark only our exact feedback as agent tests."""
import json,fcntl,requests
from .common import *
from functional_v1.common import EXP as BUDGET

def main():
 http=read(EXP/'demo/http-integration.json');model_ids=set(read(EXP/'development-selection.json')['models']);rid='d95cfb5a88b646b1b7bee4f1cb764ae4';root=BUDGET/'demo/requests';first=read(root/rid/'request.json');assert first['status']=='executed' and first['model_id']=='functional-v22-quality-20261205' and first['instruction']=='Please, add brown tone'
 expected=next(r for r in http['checks'] if r['model_id']==first['model_id'] and r['case_id']=='historical-case12-brown');assert sha(root/rid/'output.png')==sha(EXP/'demo/browser-download.png')==expected['output_sha256']
 later=[]
 for f in root.glob('*/request.json'):
  r=read(f)
  if r['model_id']=='functional-v22-quality-20261206' and r['instruction']=='大幅提高亮度。' and r['created_at']>first['created_at'] and r['status']=='declined' and not r.get('source'):later.append((f,r))
 assert len(later)==1;f,night=later[0];assert night['reason_code']=='unreliable_quality' and not f.with_name('output.png').exists();assert requests.get('http://127.0.0.1:12346/files/'+night['request_id']+'/output.png',timeout=10).status_code==404
 for fp,r in [(root/rid/'request.json',first),(f,night)]:assert not r['training_use'];r['source']='agent_round22_browser_test_not_user_feedback';write(fp,r)
 path=BUDGET/'demo/feedback.jsonl';test_ids={rid}|{r['request_id'] for r in http['checks']};agent=[]
 with path.open('r+') as handle:
  fcntl.flock(handle,fcntl.LOCK_EX);records=list(map(json.loads,handle.read().splitlines()))
  for row in records:
   if row['request_id'] in test_ids:
    assert not row['training_use'] and row['comment'].startswith('Agent R22');row['source']='agent_round22_browser_test_not_user_feedback' if row['request_id']==rid else 'agent_round22_HTTP_test_not_user_feedback';agent.append(row)
  handle.seek(0);handle.write(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in records));handle.truncate();handle.flush()
 independent=[r for r in records if read(root/r['request_id']/'request.json')['model_id'] in model_ids and not r.get('source','').startswith('agent_')];assert len(agent)==3
 files={str(f.relative_to(EXP)):sha(f) for f in (EXP/'demo').glob('browser-*.jpg')};assert len(files)==3
 write(EXP/'demo/browser-integration.json',dict(at=stamp(),status='actual_upload_edit_download_switch_refusal_feedback_verified',requests=[dict(request_id=first['request_id'],model_id=first['model_id'],status='executed',download_sha256=expected['output_sha256'],case='historical-case12'),dict(request_id=night['request_id'],model_id=night['model_id'],status='declined',reason_code=night['reason_code'],no_result=True,image_route_404=True)],screenshots=files,model_switch_cleared_old_result=True,training_use=False,agent_feedback_count=len(agent),independent_user_feedback=len(independent),quality_score=False))
 write(EXP/'demo/agent-feedback-records.json',dict(records=agent,independent_user_feedback=len(independent),training_use=False));print('Verified2 browser requests, actual download,3 agent feedback; independent',len(independent))
if __name__=='__main__':main()

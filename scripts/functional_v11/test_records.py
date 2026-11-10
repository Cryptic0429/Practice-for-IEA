"""CPU checks of real admitted/reviewed records and sealed acceptance routes."""
from functional_v1 import web
from functional_v1.common import ROOT,read,write,sha,stamp
from .web_records import EXP,DATA,records


def main():
    client=web.app.test_client()
    admission=read(DATA/'admission.json')
    assert admission['semantic_cases']==672
    result=client.get('/api/v11/experiment')
    assert result.status_code==200 and result.json['admission']==admission
    page=client.get('/v11');assert page.status_code==200
    html=page.get_data(as_text=True)
    assert '672' in html and '1344' in html and '第十一轮' in html
    assert '44例' in html and '20例' in html and 'parent-editor-replay' in html and '无新推理' in html
    training=records('training');assert len(training)==672
    assert sum(r['domain']=='portrait' for r in training)==337
    assert sum(r['domain']=='landscape' for r in training)==335
    declined=next(r for r in training if r['action']=='decline')
    assert declined['after'] is None and client.get(declined['before']).status_code==200
    assert client.get('/v11/training-image/'+declined['id']+'/after').status_code==404
    passed=client.get('/v11?kind=evaluation&model=parent-editor-replay&judgment=fail')
    assert passed.status_code==200
    assert '未合格' in passed.get_data(as_text=True)
    for path in ('/v11/evaluation-image/native-acceptance/language-merger-e1/native-accept-001/zh/before',
                 '/v11?kind=acceptance'):
        assert client.get(path).status_code==404
    write(EXP/'records-contract-check.json',dict(at=stamp(),status='passed',
        actual_training_records=672,case_domains=dict(portrait=337,landscape=335),quality_declines=44,unsupported_declines=20,
        no_declined_target_exposed=True,acceptance_routes_closed=True,
        scope='CPU Flask test client on real admitted/reviewed files; not browser or live-model integration',GPU_hours=0,
        code_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/functional_v11/web_records.py',ROOT/'scripts/functional_v11/records.html']}))
    print('R11 actual672-record UI, refusal no-image and sealed acceptance CPU checks passed.')


if __name__=='__main__':main()

"""CPU UI checks against actual admitted and reviewed files; acceptance stays closed."""
from functional_v1 import web
from functional_v1.common import ROOT,read,write,stamp,sha
from .web_records import EXP,DATA,records


def main():
    client=web.app.test_client();admission=read(DATA/'admission.json')
    overview=client.get('/api/v14/experiment')
    assert overview.status_code==200 and overview.json['admission']==admission
    html=client.get('/v14').get_data(as_text=True)
    for text in ('925','1850','154','73/100','新增照片0','第十四轮','受控重大副作用'):assert text in html,text
    labels=records('training');assert len(labels)==925
    assert sum(c['domain']=='portrait' for c in labels)==464
    assert sum(c['domain']=='landscape' for c in labels)==461
    declined=next(c for c in labels if c['action']=='decline')
    assert declined['after'] is None and client.get(declined['before']).status_code==200
    assert client.get('/v14/training-image/'+declined['id']+'/after').status_code==404
    new=next(c for c in labels if c['id']=='r13-train-native-088-new-primary')
    assert 'failed original brightening' in new['evidence']
    assert client.get(new['before']).status_code==client.get(new['after']).status_code==200
    for mid in ('visual-late4-e1',):
        for split,total in [('native-development',40),('old-native',12),('development',200)]:
            actual=records('evaluation',mid,split)
            assert len(actual)==total and all(r.get('judgment') in ('pass','fail','uncertain') for r in actual)
            for verdict in ('pass','fail','uncertain'):
                assert client.get(f'/v14?kind=evaluation&model={mid}&split={split}&judgment={verdict}').status_code==200
    for path in ('/v14?kind=acceptance','/v14/evaluation-image/native-acceptance/visual-late4-e1/native-accept-001/zh/before'):
        assert client.get(path).status_code==404
    write(EXP/'records-contract-check.json',dict(at=stamp(),status='passed',training_semantics=925,training_language_rows=1850,
        new_primary_evidence_distinguishes_failed_original_request=True,actual_reviewed_language_records_per_candidate=252,
        success_failure_uncertain_filters=True,decline_has_no_target=True,acceptance_routes_closed=True,
        scope='CPU Flask test client on actual records; not live model or browser validation',GPU_hours=0,
        code_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/functional_v14/web_records.py',ROOT/'scripts/functional_v14/records.html']}))
    print('Actual925 labels and252 reviewed language outputs per route; no declined target or acceptance exposure.')


if __name__=='__main__':main()

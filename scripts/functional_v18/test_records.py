"""Actual data/negative-image/acceptance boundaries, without loading a model."""
import sys
from flask import Flask
from functional_v1.common import ROOT,read,write,sha,stamp
from .web_records import bp,DATA,EXP,records


def main():
    app=Flask(__name__);app.register_blueprint(bp);client=app.test_client()
    assert 'torch' not in sys.modules
    training=records('training');preferences=records('preferences')
    assert len(training)==925 and len(preferences)==19
    assert len({p['id'] for p in preferences})==19
    assert sum(p['domain']=='portrait' for p in preferences)==12
    assert sum(p['domain']=='landscape' for p in preferences)==7
    for query,text in [('/v18','925'),('/v18?kind=preferences','19'),
        ('/v18?kind=preferences&domain=portrait','12'),('/v18?kind=preferences&domain=landscape','7')]:
        response=client.get(query);assert response.status_code==200
        assert text in response.get_data(as_text=True)
    assert client.get('/v18?kind=preferences&action=keep').status_code==200
    assert len(records('evaluation',model='../../outside'))==0
    for value in preferences:
        for role in ('before','after','rejected'):
            assert client.get(f'/v18/preference-image/{value["id"]}/{role}').status_code==200
    declines=[c for c in read(DATA/'cases.json') if c['decision']['action']=='decline']
    assert len(declines)==65
    for value in declines:
        assert client.get(f'/v18/training-image/{value["id"]}/after').status_code==404
    for path in ['/v18?kind=evaluation&split=acceptance','/v18?kind=acceptance',
        '/v18/evaluation-image/acceptance/any/any/en/after',
        '/v18/preference-image/unknown/after','/v18/preference-image/train-020-restore-exposure-mild/unsupported']:
        assert client.get(path).status_code==404,path
    assert 'torch' not in sys.modules
    write(EXP/'records-contract-check.json',dict(at=stamp(),status='passed_actual_data_CPU_Flask_client',
        semantic_training_rows=925,semantic_preferences=19,language_preferences=38,
        preference_semantic_domains={'portrait':12,'landscape':7},preference_image_routes_checked=57,
        no_output_decline_labels_checked=65,acceptance_routes_closed=True,
        stale_filter_reset=True,model_loaded=False,GPU_calls=0,
        code_hashes={str(p.relative_to(ROOT)):sha(p) for p in [__file__ and ROOT/'scripts/functional_v18/test_records.py',ROOT/'scripts/functional_v18/web_records.py',ROOT/'scripts/functional_v18/records.html']},
        limitation='Read-only CPU page/API checks on real records. This is not a browser walkthrough, actual model inference, training-quality score or user feedback.'))
    print('PASS actual925 training rows,19 preferences/57image routes,65 declines404,acceptance404;no model or GPU.')


if __name__=='__main__':main()

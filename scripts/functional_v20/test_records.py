"""CPU page/image boundaries; target labels are distinct from unreviewed predictions."""
import sys
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.web import app
from .web_records import records,EXP,DATA

def main():
    assert 'torch' not in sys.modules
    client=app.test_client();rows=records('training');assert len(rows)==925
    assert client.get('/v20').status_code==200 and client.get('/v20?kind=preferences').status_code==404
    assert len([c for c in rows if c['action']=='decline'])==65
    for c in read(DATA/'cases.json'):
        if c['action']=='decline':assert client.get(f'/v20/training-image/{c["id"]}/after').status_code==404
    for c in read(DATA/'cases.json')[:3]:
        assert client.get(f'/v20/training-image/{c["id"]}/before').status_code==200
        if c.get('target'):assert client.get(f'/v20/training-image/{c["id"]}/after').status_code==200
    assert client.get('/v20/evaluation-image/native-acceptance/last-e20/native-accept-001/zh/before').status_code==404
    assert client.get('/v20/evaluation-image/native-development/last-e20/native-dev-001/zh/after').status_code==404
    assert client.get('/v20?kind=evaluation&action=decline').status_code==200
    assert 'torch' not in sys.modules
    write(EXP/'records-contract-check.json',dict(at=stamp(),status='passed',training_semantics=925,decline_no_result_routes=65,
        acceptance_closed=True,unreviewed_model_images_closed=True,no_torch_imported=True,GPU_calls=0,
        code_hashes={str(ROOT/'scripts/functional_v20/web_records.py'):sha(ROOT/'scripts/functional_v20/web_records.py'),str(ROOT/'scripts/functional_v20/records.html'):sha(ROOT/'scripts/functional_v20/records.html')}))
    print('CPU records925labels,65decline after404,unreviewed/acceptance404; no torch or GPU.')

if __name__=='__main__':main()

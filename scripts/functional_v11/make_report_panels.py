"""Presentation copies from actual frozen predictions, never new visual grades."""
import os
from functional_v1.common import ROOT,read,write,sha,stamp
from .runtime import EXP
from .prepare_labels import panel
from .outcomes import EDITOR_VERSION
from .evaluate import KEYS


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    folder=EXP/'report-panels';assert not (folder/'manifest.json').exists()
    item=read(EXP/'evaluation-protocol.json')['case_files'][KEYS['native-development']]
    assert sha(ROOT/item['path'])==item['sha256']
    cases={c['id']:c for c in read(ROOT/item['path'])}
    base=EXP/'evaluation'/EDITOR_VERSION/'native-development/language-merger-e1'
    predictions=[__import__('json').loads(l) for l in (base/'predictions.jsonl').read_text().splitlines()]
    result=[]
    for number in ('001','006','008','011','023','070','100','108','161'):
        cid='native-dev-'+number;c=cases[cid]
        p=next(r for r in predictions if r['case_id']==cid and r['language']=='zh')
        assert sha(ROOT/c['image'])==c['input_sha256']
        sources=[('ORIGINAL INPUT',ROOT/c['image'])]
        if p['status']=='executed':
            assert sha(ROOT/p['output_image'])==p['output_sha256']
            sources.append(('ACTUAL R11 MODEL OUTPUT',ROOT/p['output_image']))
        title=c['instructions']['en']
        if p['status']!='executed':title+=' | '+p['status'].upper()+': NO RESULT IMAGE'
        sheet=panel(cid,title,sources,folder/f'{cid}.jpg')
        result.append(dict(case_id=cid,family=c['family'],input=c['image'],output=p['output_image'],
            status=p['status'],error=p.get('error'),panel=sheet,source_prediction_sha256=sha(base/'predictions.jsonl')))
    write(folder/'manifest.json',dict(at=stamp(),items=result,rule='Equal-scale full frames; presentation thumbnails only, not original-pixel QA or a new assessment. Failed/declined decisions have no result image.',GPU_hours=0))


if __name__=='__main__':main()

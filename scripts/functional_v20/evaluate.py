"""Execute CPU heads on exact cached allowed-input features, no 7B reruns."""
import argparse
import numpy as np
import torch
from .common import ROOT, EXP, ROUTES, protocol, read, write, sha, append, stamp
from .controller import load_controller, predict_cached
from functional_v15.outcomes import execute, EDITOR_VERSION, SCHEMA_VERSION
from functional_v1.editor import load_rgb, save

KEYS = {'old-native':'old_native', 'native-development':'native_development',
        'decision-development':'decision_development', 'development':'controlled_development'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--splits',nargs='+',choices=KEYS,default=['old-native'])
    args=ap.parse_args()
    torch.set_num_threads(4)
    p=protocol()
    net,item=load_controller(ROOT/args.checkpoint)
    assert item['protocol_sha256']==sha(EXP/'training-protocol.json')
    for split in args.splits:
        source=p['feature_sources'][KEYS[split]]
        assert sha(ROOT/source['path'])==source['sha256']
        cases=read(ROOT/source['path'])
        manifest=read(EXP/'features'/f'{KEYS[split]}-manifest.json')
        assert manifest['source_sha256']==source['sha256']
        features={(r['case_id'],r['language']):r for r in manifest['rows']}
        out=EXP/'evaluation'/EDITOR_VERSION/split/args.model
        out.mkdir(parents=True,exist_ok=False)
        write(out/'identity.json',dict(model_id=args.model,route=item['route'],controller=args.checkpoint,
            controller_sha256=sha(ROOT/args.checkpoint),runtime='functional_v20.runtime',editor=EDITOR_VERSION,
            schema=SCHEMA_VERSION,cases_sha256=source['sha256'],feature_manifest_sha256=sha(EXP/'features'/f'{KEYS[split]}-manifest.json'),
            parent_sha256=item['parent_sha256'],protocol_sha256=item['protocol_sha256'],CPU_cached_inference=True))
        rows=[]
        for case in cases:
            assert sha(ROOT/case['image'])==case['input_sha256']
            for language in ('zh','en'):
                row=dict(case_id=case['id'],group_id=case['group_id'],family=case['family'],domain=case['domain'],
                    language=language,instruction=case['instructions'][language],model_id=args.model,
                    input_image=case['image'],input_sha256=case['input_sha256'],status='failed',output_image=None,output_sha256=None)
                try:
                    f=features[case['id'],language]
                    assert sha(ROOT/f['path'])==f['sha256']
                    cached=torch.load(ROOT/f['path'],map_location='cpu',weights_only=True)
                    assert cached['input_sha256']==case['input_sha256'] and cached['request']==case['instructions'][language]
                    assert cached['protocol_sha256']==item['protocol_sha256']
                    response,decision=predict_cached(net,item,cached)
                    result=execute(load_rgb(ROOT/case['image']),decision)
                    row.update(response=response,decision=decision,parameters=result['parameters'],execution_status=result['status'])
                    if result['status']=='declined':
                        assert result['result'] is None
                        row.update(status='declined',reason_code=result['reason_code'])
                    else:
                        path=out/'images'/f'{case["id"]}-{language}.png'
                        save(path,result['result'])
                        row.update(status='executed',output_image=str(path.relative_to(ROOT)),output_sha256=sha(path),
                            input_output_identical=bool(np.array_equal(load_rgb(ROOT/case['image']),result['result'])))
                except Exception as error: row['error']=f'{type(error).__name__}: {error}'
                row['at']=stamp();append(out/'predictions.jsonl',row);rows.append(row)
        write(out/'status.json',dict(status='completed_predictions',completed=len(rows),total=2*len(cases),
              status_counts={s:sum(r['status']==s for r in rows) for s in ('executed','declined','failed')},at=stamp()))
    assert not torch.cuda.is_initialized()

if __name__=='__main__':main()

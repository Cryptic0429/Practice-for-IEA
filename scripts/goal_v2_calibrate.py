"""CPU-only validation diagnostics; never accesses sealed test rows."""
import argparse, hashlib, json, statistics, types
from pathlib import Path
from evaluation_core import Scorer, sample_seed
ROOT=Path('/mnt/sdc/zhc/IEA')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--predictions',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    source=Path(a.predictions);predictions=[json.loads(l) for l in source.open()]
    validation=json.loads((ROOT/'data/goal-v2-20261103/validation.json').read_text());lookup={r['id']:r for r in validation}
    assert predictions and len({r['id'] for r in predictions})==len(predictions)
    assert all(r['id'] in lookup for r in predictions), 'Validation only'
    out=ROOT/'outputs/goal-v2-20261103/calibration'/a.name;out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((ROOT/'configs/evaluation-protocol-v1.json').read_text());s=Scorer(ROOT,protocol);np=s.np
    records=[]
    for pred in predictions:
        row=lookup[pred['id']];original=s.load(row['images'][0]);ref=s.load(row['extra_info']['reference_image_path'])
        editor=s.editor(row['images'][0]);params=pred['parsing'].get('parameters')
        def execute(p):
            np.random.seed(sample_seed(row['id'],protocol['seed']));return editor.process_image(p)
        edited=execute(params) if pred['parsing']['valid'] else original
        values={'NoEdit':s.diff(original,ref)['pixel_L1'],'original':s.diff(edited,ref)['pixel_L1']}
        for alpha in [.25,.5,.75]:
            blend=np.rint((1-alpha)*original.astype(np.float32)+alpha*edited.astype(np.float32)).clip(0,255).astype(np.uint8)
            values[f'blend-{alpha}']=s.diff(blend,ref)['pixel_L1']
        if params is not None and pred['parsing']['valid']:
            for key in ['brightness','contrast','temperature']:
                changed=dict(params);changed[key]=0;values[f'drop-{key}']=s.diff(execute(changed),ref)['pixel_L1']
        else:
            for key in ['brightness','contrast','temperature']:values[f'drop-{key}']=values['NoEdit']
        values['label-original']=s.diff(execute(row['extra_info']['label']),ref)['pixel_L1']
        def contrast(self,image,value):return .5+(1+max(-100,min(100,value))/100.)*(image-.5)
        editor.adjust_contrast=types.MethodType(contrast,editor)
        values['continuous-contrast']=s.diff(execute(params),ref)['pixel_L1'] if pred['parsing']['valid'] else values['NoEdit']
        values['label-continuous-contrast']=s.diff(execute(row['extra_info']['label']),ref)['pixel_L1']
        records.append({'id':row['id'],'group_id':row['group_id'],'metrics':values})
    summary={'status':'completed','n':len(records),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
             'mean_l1':{k:statistics.mean(r['metrics'][k] for r in records) for k in records[0]['metrics']},
             'limits':['Validation-only exploratory diagnostics; not a model improvement claim.','One globally fixed blend coefficient per condition, no per-image reference-based tuning.','Edited pixels with failed outputs fall back to original and remain included.']}
    (out/'rows.json').write_text(json.dumps(records,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()

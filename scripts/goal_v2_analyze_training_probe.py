"""Separate memorization ability from held-out performance on a fixed training cohort."""
import hashlib,json,statistics
from pathlib import Path
from evaluation_core import Scorer,parse_parameters
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def main():
    source=ROOT/'data/goal-v2-20261103/train-memorize64.json';rows=json.loads(source.read_text());ids=[r['id'] for r in rows]
    assert len(rows)==len(set(ids))==len({r['group_id'] for r in rows})==64
    protocol=json.loads((ROOT/'configs/evaluation-protocol-v1.json').read_text());scorer=Scorer(ROOT,protocol)
    oracle=[]
    for row in rows:
        parsed=parse_parameters('```json\n'+json.dumps(row['extra_info']['label'])+'\n```',protocol)
        record=scorer.score(row,parsed,None);assert record['editor_success'] and not record['metric_error']
        oracle.append(record['edited_metrics']['pixel_L1'])
    names=sorted(p.name.removeprefix('train_probe-') for p in (OUT/'training-probe-evaluation').glob('train_probe-*'));models={}
    targets={r['id']:r['extra_info']['label'] for r in rows}
    for name in names:
        run=OUT/'training-probe-evaluation'/f'train_probe-{name}'
        if not (run/'status.json').exists() or json.loads((run/'status.json').read_text())['status']!='completed':continue
        p=run/'predictions.jsonl';records=[json.loads(l) for l in p.read_text().splitlines()];byid={r['id']:r for r in records}
        assert len(records)==len(byid)==64 and set(byid)==set(ids)
        records=[byid[i] for i in ids];valid=[r for r in records if r['parsing']['valid']]
        exact=0;contrast_error=[];mae=[]
        for rec in valid:
            label=targets[rec['id']];pred=rec['parsing']['parameters']
            errors=[abs(pred.get(k,0)-label.get(k,0)) for k in protocol['parser']['allowed_tools']]
            exact+=max(errors)<1e-6;mae.append(statistics.mean(errors));contrast_error.append(abs(pred.get('contrast',0)-label.get('contrast',0)))
        models[name]={'n':64,'mean_l1':statistics.mean(r['scoring']['edited_metrics']['pixel_L1'] for r in records),
            'NoEdit_l1':statistics.mean(r['scoring']['baseline_metrics']['pixel_L1'] for r in records),
            'executable':sum(r['scoring']['editor_success'] for r in records),'exact_dense_parameter_vectors':exact,
            'parameter_mae16_valid_only':statistics.mean(mae) if mae else None,
            'contrast_mae_valid_only':statistics.mean(contrast_error) if contrast_error else None,
            'gap_to_label_execution':statistics.mean(r['scoring']['edited_metrics']['pixel_L1'] for r in records)-statistics.mean(oracle),
            'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        scope=OUT/name/'memorization-scope.json'
        if scope.exists():models[name]['training_scope']=json.loads(scope.read_text())
    report={'status':'completed_probes_only','independent_training_groups':64,
        'initial_probe_training_exposures':512,'initial_probe_training_epochs':8,'label_execution_l1':statistics.mean(oracle),'models':models,
        'limits':['Training memorization diagnostic; cannot establish held-out generalization or satisfy the goal.','All models receive only original image and original user editing request plus identical JSON-only schema; no reference or label input.',
                  'Parameter MAE among valid outputs is secondary and must be read with executable coverage; image L1 retains every failure.']}
    dest=OUT/'training-probe-analysis';dest.mkdir(exist_ok=True);tmp=dest/'summary.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(dest/'summary.json')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

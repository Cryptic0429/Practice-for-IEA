"""Audit training-label image benefit and explicit global instruction directions, CPU only."""
import hashlib,json
from pathlib import Path
import cv2
import numpy as np
from evaluation_core import Scorer,parse_parameters
from goal_v2_analyze_validation import directions,features,THRESHOLD
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103/training-label-audit'
def main():
    assert not OUT.exists();OUT.mkdir()
    cv2.setNumThreads(2)
    data=ROOT/'data/goal-v2-20261103/train.json';rows=json.loads(data.read_text())
    manifest=json.loads((data.parent/'manifest.json').read_text())
    assert hashlib.sha256(data.read_bytes()).hexdigest()==manifest['files']['train']['sha256']
    protocol=json.loads((ROOT/'configs/evaluation-protocol-v1.json').read_text());scorer=Scorer(ROOT,protocol)
    records=[]
    for row in rows:
        label=row['extra_info']['label'];parsed=parse_parameters('```json\n'+json.dumps(label)+'\n```',protocol)
        assert parsed['valid']
        score=scorer.score(row,parsed,OUT/'edited'/f'{hashlib.sha256(row["id"].encode()).hexdigest()[:20]}.png')
        assert score['editor_success'] and not score['metric_error']
        expected=directions(row['extra_info']['instruction']);checks={}
        if expected:
            original=cv2.imread(row['images'][0]);edited=cv2.imread(score['edited_image_path']);f0=features(original);f1=features(edited)
            for feature,sign in expected.items():
                delta=sign*(f1[feature]-f0[feature]);checks[feature]={'signed_delta':delta,'same_direction':delta>THRESHOLD,'opposite_direction':delta<-THRESHOLD}
        record={'id':row['id'],'group_id':row['group_id'],'instruction_type':row['extra_info']['inst_type'],
                'NoEdit_l1':score['baseline_metrics']['pixel_L1'],'label_l1':score['edited_metrics']['pixel_L1'],
                'l1_improvement':score['l1_improvement'],'direction_checks':checks,
                'clear_direction_conflict':any(v['opposite_direction'] for v in checks.values())}
        records.append(record)
        with (OUT/'rows.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        (OUT/'progress.json').write_text(json.dumps({'status':'running','completed':len(records),'total':len(rows)})+'\n')
    result={'status':'completed','n':len(records),'source_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
       'mean_NoEdit_l1':float(np.mean([r['NoEdit_l1'] for r in records])),
       'mean_label_l1':float(np.mean([r['label_l1'] for r in records])),
       'labels_improving_reference':sum(r['l1_improvement']>1e-6 for r in records),
       'direction_assessable':sum(bool(r['direction_checks']) for r in records),
       'clear_direction_conflicts':sum(r['clear_direction_conflict'] for r in records),
       'positive_reference_gain_without_clear_direction_conflict':sum(r['l1_improvement']>1e-6 and not r['clear_direction_conflict'] for r in records),
       'limits':['Training-only data quality audit, not a model evaluation.','Global direction proxy cannot determine full semantic correctness or local edits.','No filtering or relabeling applied to ongoing pilots.']}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(OUT/'progress.json').write_text(json.dumps({'status':'completed','completed':len(rows),'total':len(rows)})+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

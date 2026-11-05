"""CPU-only near-duplicate screening. No model or test editing/reference scores."""
import hashlib,json
from pathlib import Path
import cv2
import numpy as np
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103/holdout-image-audit'
def fingerprint(path):
    image=cv2.imread(path);assert image is not None,path
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    small=cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
    dct=cv2.dct(small)[:8,:8].reshape(-1)[1:];bits=dct>np.median(dct)
    h=sum(int(b)<<i for i,b in enumerate(bits))
    grid=cv2.resize(gray,(64,64),interpolation=cv2.INTER_AREA).astype(np.float32)/255.
    return h,grid,float(grid.std())
def main():
    OUT.mkdir(exist_ok=False);cv2.setNumThreads(2)
    data=ROOT/'data/goal-v2-20261103/test_sealed.json';test=json.loads(data.read_text());testgroups={r['group_id'] for r in test}
    groups=json.loads((ROOT/'data/stage1/splits/image-edit-3k-v1/group-assignments.json').read_text())
    # Conservatively screen against every other source group, not only the selected training sample IDs.
    development={g['group_id']:g['source_paths'][0] for g in groups if g['group_id'] not in testgroups}
    features={};records=[]
    for gid,path in sorted(development.items()):
        features[gid]=fingerprint(path)
    for row in test:
        h,a,std=fingerprint(row['images'][0]);matches=[]
        for gid,(other,b,otherstd) in features.items():
            distance=(h^other).bit_count()
            if distance>6:continue
            mae=float(np.abs(a-b).mean())
            corr=float(np.corrcoef(a.reshape(-1),b.reshape(-1))[0,1]) if min(std,otherstd)>.005 else None
            if (corr is not None and corr>.985) or mae<.01:
                matches.append({'development_group':gid,'phash_hamming':distance,'gray_64_mae':mae,'gray_64_correlation':corr})
        records.append({'id':row['id'],'group_id':row['group_id'],'low_variance':std<.005,'matches':matches})
        (OUT/'progress.json').write_text(json.dumps({'status':'running','completed_test_groups':len(records),'test_groups':len(test)})+'\n')
    excluded=[r['group_id'] for r in records if r['matches']]
    result={'status':'completed','test_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'test_groups':len(test),
            'screened_development_groups':len(development),'flagged_groups':len(excluded),'excluded_from_confirmatory_test':excluded,
            'eligible_groups':sorted(testgroups-set(excluded)),
            'rule':'pHash Hamming <=6 and (grayscale64 correlation >0.985 or grayscale64 MAE <0.01)',
            'limits':['Conservative automated screen may exclude legitimate similar photos and miss crop/rotation/semantic near duplicates.',
                      'No model inference, user-instruction inspection or editing-to-reference test scores.',
                      'Original sealed split remains immutable; final model-lock manifest must explicitly incorporate these exclusions.'],
            'records':records}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(OUT/'progress.json').write_text(json.dumps({'status':'completed','completed_test_groups':len(test)})+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','eligible_groups','excluded_from_confirmatory_test']},indent=2))
if __name__=='__main__':main()

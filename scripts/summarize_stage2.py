"""All-row paired estimates after all frozen Stage 2 predictions are complete."""
import json
import hashlib
from pathlib import Path
import numpy as np
import cv2

ROOT=Path('/mnt/sdc/zhc/IEA')
BASE=ROOT/'outputs/stage2-A-edit-grpo-v1'


def bootstrap(values,seed=20261101):
    values=np.asarray(values,dtype=np.float64)
    rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()),median=float(np.median(values)),
                ci95=[float(x) for x in np.quantile(means,[.025,.975])],
                improved=int((values>1e-6).sum()),worsened=int((values< -1e-6).sum()),
                tied=int((np.abs(values)<=1e-6).sum()),
                trimmed_mean_10pct=float(np.sort(values)[len(values)//10:len(values)-len(values)//10].mean()))


def main():
    out=BASE/'summary.json';assert not out.exists()
    lock=json.loads((ROOT/'configs/stage2-selected-models.json').read_text());assert lock['status']=='frozen'
    results={}
    evidence={}
    for split in ['test_new','test_regression']:
        source=ROOT/'data/stage2/stage2-A-edit-grpo-v1'/f'{split}.json'
        target=json.loads(source.read_text());ids=[r['id'] for r in target]
        assert len({r['group_id'] for r in target})==len(target)
        by_model={}
        for name in ['A','SFT','GRPO']:
            folder=BASE/'evaluation'/f'{split}-{name}'
            status=json.loads((folder/'status.json').read_text())
            assert status['status']=='completed' and status['completed']==len(ids)
            run=json.loads((folder/'run.json').read_text())
            assert run['identity']['adapter_sha256']==lock['models'][name]['adapter_sha256']
            path=folder/'predictions.jsonl'
            rows=[json.loads(x) for x in path.read_text().splitlines()]
            assert [r['id'] for r in rows]==ids
            assert all(r['scoring'].get('metric_error') is None for r in rows)
            evidence[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            by_model[name]=rows
        baseline=np.array([r['scoring']['baseline_metrics']['pixel_L1'] for r in by_model['A']])
        baseline_rmse=np.array([r['scoring']['baseline_metrics']['pixel_L2'] for r in by_model['A']])
        l1={'NoEdit':baseline}
        rmse={'NoEdit':baseline_rmse}
        metrics={'NoEdit':dict(mean_l1=float(baseline.mean()),mean_rmse=float(baseline_rmse.mean()),
                               rows=len(ids),exact_identity_images=len(ids))}
        for name,rows in by_model.items():
            assert np.array_equal(baseline,[r['scoring']['baseline_metrics']['pixel_L1'] for r in rows])
            distances=np.array([r['scoring']['edited_metrics']['pixel_L1'] for r in rows]);assert np.isfinite(distances).all()
            l1[name]=distances
            rmse[name]=np.array([r['scoring']['edited_metrics']['pixel_L2'] for r in rows])
            assert np.isfinite(rmse[name]).all()
            assert np.array_equal(baseline_rmse,[r['scoring']['baseline_metrics']['pixel_L2'] for r in rows])
            valid=np.array([r['scoring']['editor_success'] for r in rows])
            unchanged=0
            for row in rows:
                original=cv2.imread(row['input_image']);edited=cv2.imread(row['scoring']['edited_image_path'])
                assert original is not None and edited is not None
                unchanged+=int(np.array_equal(original,edited))
            metrics[name]=dict(rows=len(rows),mean_l1=float(distances.mean()),
                mean_rmse=float(rmse[name].mean()),
                executable=int(valid.sum()),executable_fraction=float(valid.mean()),
                success_only_l1=float(distances[valid].mean()) if valid.any() else None,
                empty_parameters=sum(r['empty_parameters'] for r in rows),
                all_zero_parameters=sum(r['all_zero_parameters'] for r in rows),
                exact_identity_images=unchanged,baseline_gain=bootstrap(baseline-distances),
                truncations=sum(r['finish_reason']=='length' for r in rows))
        comparisons={}
        for candidate,reference in [('GRPO','A'),('GRPO','SFT'),('GRPO','NoEdit'),('SFT','A'),('SFT','NoEdit')]:
            gain=l1[reference]-l1[candidate]
            result=bootstrap(gain)
            result['rmse_gain']=bootstrap(rmse[reference]-rmse[candidate])
            result['by_instruction_type']={}
            for kind in sorted({r['extra_info']['inst_type'] for r in target}):
                subset=np.array([r['extra_info']['inst_type']==kind for r in target])
                result['by_instruction_type'][kind]=dict(n=int(subset.sum()),**bootstrap(gain[subset]))
            # Describe concentration without replacing the main all-row result.
            top=np.sort(gain)[-5:]
            result['five_largest_gains_sum']=float(top.sum())
            result['mean_without_five_largest_gains']=float(np.sort(gain)[:-5].mean())
            comparisons[f'{candidate}_vs_{reference}']=result
        results[split]=dict(metrics=metrics,comparisons=comparisons)
    report=dict(status='complete',results=results,prediction_sha256=evidence,
        uncertainty='Paired bootstrap,10000 resamples,one original-image group per row; exploratory,unadjusted95% intervals.',
        interpretation='New100 is the final held-out test within recorded project history; old200 is previously inspected regression.',
        limits=['One training seed','L1/RMSE are not direct human instruction satisfaction','No perceptual near-duplicate certification',
                'Additional SFT and GRPO have equal updates but different exposure and computation budgets'])
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()

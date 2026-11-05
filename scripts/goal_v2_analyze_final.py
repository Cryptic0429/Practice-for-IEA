"""All71 groups, all locked models and seeds; replay saved pixels and paired uncertainty."""
import hashlib,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from goal_v2_analyze_validation import directions,features,THRESHOLD
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def init_worker(predictions):
    global RECORDS,load_image,diff
    import sys
    sys.path.insert(0,str(ROOT/'repo'));from src.utils import load_and_resize_image,compute_diff_score
    RECORDS=predictions;load_image=load_and_resize_image;diff=compute_diff_score;cv2.setNumThreads(1)
def image_row(row):
    original=load_image(row['images'][0]);reference=load_image(row['extra_info']['reference_image_path']);baseline=diff(original,reference);f0=features(original);wanted=directions(row['extra_info']['instruction']);results={}
    for name,lookup in RECORDS.items():
        rec=lookup[row['id']];sc=rec['scoring'];path=Path(sc['edited_image_path']);assert sha(path)==rec['edited_image_sha256']
        edited=cv2.imread(str(path));assert edited is not None and edited.shape==original.shape
        replay=diff(edited,reference);error=abs(replay['pixel_L1']-sc['edited_metrics']['pixel_L1']);assert error<1e-7
        assert abs(baseline['pixel_L1']-sc['baseline_metrics']['pixel_L1'])<1e-7
        f1=features(edited);edit=float(np.abs(edited.astype(np.float32)-original.astype(np.float32)).mean()/255.)
        checks={k:{'expected_sign':s,'delta':f1[k]-f0[k],'classification':'same_direction' if s*(f1[k]-f0[k])>THRESHOLD else ('opposite_direction' if s*(f1[k]-f0[k])<-THRESHOLD else 'no_clear_change')} for k,s in wanted.items()}
        results[name]={'edit_l1':edit,'pixel_identical':bool(np.array_equal(original,edited)),'near_identity':edit<1/255,'direction_checks':checks,'metric_replay_absdiff':error}
    return {'id':row['id'],'group_id':row['group_id'],'models':results}
def main():
    lockfile=OUT/'final-models-lock.json';lock=json.loads(lockfile.read_text());assert lock['status']=='frozen_for_final_test'
    protocol=json.loads((OUT/'final-protocol.json').read_text());assert sha(OUT/'final-protocol.json')==lock['final_protocol_sha256']
    assert sha(Path(protocol['test_source']['path']))==lock['test_source_sha256'];rows=json.loads(Path(protocol['test_source']['path']).read_text());ids=[r['id'] for r in rows];assert len(ids)==len(set(ids))==len({r['group_id'] for r in rows})==71
    opened=json.loads((OUT/'test-opened.json').read_text());assert opened['model_lock_sha256']==sha(lockfile)
    names=lock['baseline_keys']+lock['candidate_keys'];assert len(names)==7
    predictions={};digests={};l1={};noedit=None
    for name in names:
        directory=OUT/'final-evaluation'/name;assert json.loads((directory/'status.json').read_text())['status']=='completed'
        identity=json.loads((directory/'run.json').read_text())['identity'];assert identity['model_lock_sha256']==sha(lockfile) and identity['adapter_sha256']==lock['models'][name]['adapter_sha256']
        records=[json.loads(l) for l in (directory/'predictions.jsonl').open()];assert [r['id'] for r in records]==ids and all(r['group_id']==row['group_id'] for r,row in zip(records,rows))
        assert len(records)==71;predictions[name]={r['id']:r for r in records};digests[name]=sha(directory/'predictions.jsonl');l1[name]=np.array([r['scoring']['edited_metrics']['pixel_L1'] for r in records]);assert np.isfinite(l1[name]).all()
        this=np.array([r['scoring']['baseline_metrics']['pixel_L1'] for r in records]);noedit=this if noedit is None else noedit;assert np.array_equal(this,noedit)
    for row in rows:
        assert len({(predictions[n][row['id']]['input_sha256'],predictions[n][row['id']]['reference_sha256']) for n in names})==1
    output=OUT/'final-analysis';output.mkdir(exist_ok=True);cache=output/'image-diagnostics.json'
    if cache.exists():
        cached=json.loads(cache.read_text());assert cached['predictions_sha256']==digests;diagnostics=cached['rows']
    else:
        with ProcessPoolExecutor(max_workers=8,initializer=init_worker,initargs=(predictions,)) as pool:diagnostics=list(pool.map(image_row,rows))
        cache.write_text(json.dumps({'predictions_sha256':digests,'rows':diagnostics,'scope':'Original/edited saved-pixel replay with original frozen diff function; global direction proxies only.'},indent=2)+'\n')
    def diagnostics_for(name):return [r['models'][name] for r in diagnostics]
    summaries={}
    for name in names:
        records=[predictions[name][i] for i in ids];d=diagnostics_for(name);assessed=[r for r in d if r['direction_checks']]
        summaries[name]={'n':71,'mean_l1':float(l1[name].mean()),'NoEdit_l1':float(noedit.mean()),'executable':sum(r['scoring']['editor_success'] for r in records),'fallback':sum(r['scoring']['fallback_used'] for r in records),
                         'empty':sum(r['empty_parameters'] for r in records),'zero':sum(r['all_zero_parameters'] for r in records),'empty_or_zero':sum(r['empty_parameters'] or r['all_zero_parameters'] for r in records),
                         'pixel_identical':sum(r['pixel_identical'] for r in d),'near_identity':sum(r['near_identity'] for r in d),'mean_input_output_l1':float(np.mean([r['edit_l1'] for r in d])),
                         'edit_l1_quantiles_0_05_50_95_100':np.quantile([r['edit_l1'] for r in d],[0,.05,.5,.95,1]).tolist(),
                         'direction_assessable':len(assessed),'all_assessed_directions_aligned':sum(all(c['classification']=='same_direction' for c in r['direction_checks'].values()) for r in assessed),
                         'max_saved_pixel_metric_replay_error':max(r['metric_replay_absdiff'] for r in d),'predictions_sha256':digests[name]}
    strongest=min(lock['baseline_keys'],key=lambda n:l1[n].mean());rng=np.random.default_rng(20261103);ix=rng.integers(0,71,(10000,71));comparisons={}
    def comparison(a,b):
        delta=a-b;boot=delta[ix].mean(1);relative=boot/a[ix].mean(1)
        return {'paired_mean_gain':float(delta.mean()),'relative_mean_gain':float(delta.mean()/a.mean()),'ci95_gain':np.quantile(boot,[.025,.975]).tolist(),'ci95_relative':np.quantile(relative,[.025,.975]).tolist(),'improved':int((delta>1e-6).sum()),'worsened':int((delta<-1e-6).sum()),'ties':int((np.abs(delta)<=1e-6).sum())}
    robustness={};checks={}
    for candidate in lock['candidate_keys']:
        comparisons[candidate]={baseline:comparison(l1[baseline],l1[candidate]) for baseline in lock['baseline_keys']};comparisons[candidate]['NoEdit']=comparison(noedit,l1[candidate])
        b=summaries[strongest];c=summaries[candidate];bd=diagnostics_for(strongest);cd=diagnostics_for(candidate)
        common=np.array([predictions[strongest][i]['scoring']['editor_success'] and predictions[candidate][i]['scoring']['editor_success'] for i in ids])
        union=np.array([x['edit_l1']>=1/255 or y['edit_l1']>=1/255 for x,y in zip(bd,cd)]);both=np.array([x['edit_l1']>=1/255 and y['edit_l1']>=1/255 for x,y in zip(bd,cd)])
        def subset(mask):
            return {'n':int(mask.sum()),'paired_mean_gain':float((l1[strongest][mask]-l1[candidate][mask]).mean()) if mask.any() else None,'relative_gain':float((l1[strongest][mask]-l1[candidate][mask]).mean()/l1[strongest][mask].mean()) if mask.any() else None}
        robustness[candidate]={'common_executable':subset(common),'substantive_edit_union':subset(union),'both_substantive_edits':subset(both)}
        # Preserve all71 rows and remove only positive gains from near-no-op/fallback rows.
        # This checks whether the target still holds without credit for doing nothing.
        no_op=np.array([d['near_identity'] or predictions[candidate][i]['empty_parameters'] or predictions[candidate][i]['all_zero_parameters'] or not predictions[candidate][i]['scoring']['editor_success'] for i,d in zip(ids,cd)])
        adjusted=l1[strongest]-l1[candidate];adjusted=adjusted.copy();adjusted[no_op]=np.minimum(adjusted[no_op],0.)
        adjusted_boot=adjusted[ix].mean(1)
        robustness[candidate]['positive_noop_gain_removed']={'n':71,'affected_rows':int(no_op.sum()),'paired_mean_gain':float(adjusted.mean()),'relative_gain':float(adjusted.mean()/l1[strongest].mean()),'ci95_gain':np.quantile(adjusted_boot,[.025,.975]).tolist(),'definition':'Set positive baseline-minus-candidate gains to zero on candidate fallback, empty/zero or editL1<1/255 rows; keep negative gains and all71 rows.'}
        primary=comparisons[candidate][strongest]
        checks[candidate]={'point_target':primary['relative_mean_gain']>=.03,'positive_paired_interval':primary['ci95_gain'][0]>0,'execution_maintained':c['executable']>=b['executable'],
                           'gain_not_explained_by_noop':robustness[candidate]['positive_noop_gain_removed']['relative_gain']>=.03 and robustness[candidate]['positive_noop_gain_removed']['ci95_gain'][0]>0,
                           'common_executable_gain_remains':robustness[candidate]['common_executable']['relative_gain'] is not None and robustness[candidate]['common_executable']['relative_gain']>=.03,
                           'substantive_union_gain_remains':robustness[candidate]['substantive_edit_union']['relative_gain'] is not None and robustness[candidate]['substantive_edit_union']['relative_gain']>=.03,
                           'global_direction_noninferior':c['all_assessed_directions_aligned']>=b['all_assessed_directions_aligned'] if c['direction_assessable'] else None}
    seed_average=np.stack([l1[n] for n in lock['candidate_keys']]).mean(0);perseed=[comparisons[n][strongest]['relative_mean_gain'] for n in lock['candidate_keys']]
    numeric_pass=all(all(v is True for v in checks[n].values()) for n in checks)
    report={'status':'numerical_complete_visual_review_pending','model_lock_sha256':sha(lockfile),'test_source_sha256':lock['test_source_sha256'],'independent_source_groups':71,'training_seeds':lock['training_seeds'],
            'models':summaries,'NoEdit_l1':float(noedit.mean()),'strongest_original_baseline':strongest,'per_seed_comparisons':comparisons,'seed_average_comparison':comparison(l1[strongest],seed_average),
            'seed_relative_gain_range':[min(perseed),max(perseed)],'guardrails':checks,'subset_robustness':robustness,'all_numerical_checks_passed':numeric_pass,'goal_complete':False,
            'pending':['Anonymous30-case broader visual review and decoding','Final selected adapter/environment/configuration/full-result archive verification','Root completion audit'],
            'limits':['Intervals resample71 original source groups jointly across seeds, not213 independent seed-by-image rows.','Per-comparison95% intervals; report all frozen original baselines and each seed, with selection/multiplicity limitations.','Global direction proxies do not establish local edits or aesthetics; AI-assisted review is separately required.','No further test-driven model/recipe selection. If not confirmed, this71-group set becomes regression data.']}
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':report['status'],'strongest_original_baseline':strongest,'seed_relative_gain_range':report['seed_relative_gain_range'],'all_numerical_checks_passed':numeric_pass,'goal_complete':False},indent=2))
if __name__=='__main__':main()

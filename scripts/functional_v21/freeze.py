"""Freeze actual data, input cache and original archived parent decision provenance."""
import json,shutil
from .common import ROOT,EXP,DATA,PARENT,PARENT_SHA,TOOLS,read,write,sha,stamp,GPUJob

def main():
    assert not (EXP/'training-protocol.json').exists()
    old=read(ROOT/'experiments/functional-v20-20261107/training-protocol.json')
    src=ROOT/'data/training/functional-v20-20261107';DATA.mkdir(parents=True,exist_ok=True)
    for name in ('cases.json','train-accepted.json','training-pool.json'):shutil.copy2(src/name,DATA/name)
    cases=read(DATA/'cases.json');assert len(cases)==925 and len({c['group_id'] for c in cases})==240
    bad=[c for c in cases if c['decision'].get('reason')=='unreliable_quality']
    assert len(bad)==44 and len({c['group_id'] for c in bad})==23
    sources=dict(old['feature_sources']);sources['train']=dict(path=str((DATA/'cases.json').relative_to(ROOT)),sha256=sha(DATA/'cases.json'))
    hidden={}
    for split in ('train','old_native','native_development','decision_development'):
        path=ROOT/f'experiments/functional-v20-20261107/features/{split}-manifest.json';m=read(path)
        assert m['parent_sha256']==PARENT_SHA and m['source_sha256']==sources[split]['sha256']
        hidden[split]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
    prior=ROOT/'archives/functional-v15-20261106';receipt=read(prior/'complete-receipt.json')
    archive_manifest=prior/'complete-snapshot/MANIFEST.json';assert sha(archive_manifest)==receipt['manifest_sha256']
    archived={r['path']:r for r in read(archive_manifest)['files']}
    from functional_v15.outcomes import parse
    semantics={};provenance={}
    for split,internal in [('old-native','old_native'),('native-development','native_development'),('decision-development','decision_development'),('development','controlled_development')]:
        source=sources[internal];assert sha(ROOT/source['path'])==source['sha256']
        indexed={c['id']:c for c in read(ROOT/source['path'])}
        folder=ROOT/'experiments/functional-v15-20261106/evaluation/functional-editor-v15-shared-shadow-gain-candidate'/split/'language-merger-e1'
        for name in ('identity.json','predictions.jsonl'):
            path=folder/name;assert sha(path)==archived[str(path.relative_to(ROOT))]['sha256']
        identity=read(folder/'identity.json')
        assert identity['adapter_sha256']==PARENT_SHA and identity['runtime']=='functional_v15.runtime' and identity['cases_sha256']==source['sha256']
        rows=[]
        for row in map(json.loads,(folder/'predictions.jsonl').read_text().splitlines()):
            c=indexed[row['case_id']]
            assert row['input_sha256']==c['input_sha256'] and row['instruction']==c['instructions'][row['language']]
            if row['status']!='failed':assert parse(row['response'])==row['decision']
            rows.append({k:row[k] for k in ('case_id','language','input_sha256','instruction','status','response') if k in row}|
                dict(parent_decision=row.get('decision'),parent_error=row.get('error')))
        assert len(rows)==len({(r['case_id'],r['language']) for r in rows})==2*len(indexed)
        semantics[internal]=rows
        provenance[internal]=dict(identity_path=str((folder/'identity.json').relative_to(ROOT)),identity_sha256=sha(folder/'identity.json'),
            predictions_path=str((folder/'predictions.jsonl').relative_to(ROOT)),predictions_sha256=sha(folder/'predictions.jsonl'),
            source_archive=receipt['archive'],source_archive_sha256=receipt['sha256'],manifest_sha256=receipt['manifest_sha256'])
    path=EXP/'parent-semantics.json';write(path,dict(at=stamp(),records=semantics,provenance=provenance,
        purpose='Reuse only exact original input-bound R15 decisions, never its images, labels or visual scores. No new parent inference claimed.'))
    groups={c['group_id'] for c in cases}
    for split,source in sources.items():
        if split!='train':assert not groups & {c['group_id'] for c in read(ROOT/source['path'])}
    previous_eval=read(ROOT/'experiments/functional-v19-20261107/evaluation-protocol.json')
    sealed={key:previous_eval['case_files'][key] for key in ('native_acceptance','controlled_acceptance')}
    for source in sealed.values():
        assert sha(ROOT/source['path'])==source['sha256']
        assert not groups & {c['group_id'] for c in read(ROOT/source['path'])}
    assert sha(PARENT/'adapter_model.safetensors')==PARENT_SHA
    ledger=read(ROOT/'experiments/functional-v1-20261104/resources.json');used=GPUJob.consumed(ledger)
    assert used<23.65
    hashes=dict(old['code_hashes'])
    for path in (ROOT/'scripts/functional_v21').glob('*.py'):hashes[str(path.relative_to(ROOT))]=sha(path)
    value=dict(at=stamp(),status='frozen_before_texture_features_head_training_or_new_functional_outputs',
        hypothesis='R20 can improve old edits but its standalone semantic head omits tools and still amplifies noise. Preserve frozen R15 actions/tools/signs, learn only unsigned magnitudes plus a quality veto using current-input residual observations.',
        policies=['magnitude','magnitude-quality'],policy_comparison='Both policies use exactly the same trained head weights and parent decisions. Only applying the learned risk>=.5 veto differs; these are not two independent trainings.',
        parent=str(PARENT.relative_to(ROOT)),parent_sha256=PARENT_SHA,parent_identity='Actual frozen R15e1, earlier-start-named directory',
        sources=sources,sealed_acceptance_sources=sealed,hidden_manifests=hidden,hidden_protocol_sha256=sha(ROOT/'experiments/functional-v20-20261107/training-protocol.json'),
        parent_semantics=dict(path=str((EXP/'parent-semantics.json').relative_to(ROOT)),sha256=sha(EXP/'parent-semantics.json')),
        data=dict(semantics=925,language_records=1850,groups=240,portrait_groups=120,landscape_groups=120,original_English=154,AI_authored=771,
            edit=780,keep=80,quality_decline=44,quality_decline_groups=23,unsupported_decline=21,new_photos=0,new_requests=0,new_labels=0),
        features='R20 exact frozen R15 last hidden3584+16RGBstats, plus12 fixed512thumb median-luminance/chroma residual observations. Natural texture also produces residuals; no feature is an automatic quality verdict. No target/request/caseID in residual function.',
        training=dict(epochs=80,checkpoints=[20,80],steps_per_epoch=58,seed=20261205,replicate_seed=20261206,paired_semantics_batch=16,
            language_inputs_per_batch=32,rows_per_epoch=1850,learning_rate=.001,weight_decay=.01,CPU_only=True,
            architecture='3612->128GELU/dropout.1->9sigmoid unsigned strengths+1risk logit',
            loss='4 active-only unsigned SmoothL1(beta.1) + supported-only weighted risk BCE(sqrt780/44 cap5) +.25 paired magnitudeMSE and risk BernoulliJS.',
            quality_scope='780 edits are reliable positives and44 quality-declines unreliable; keep80 andunsupported21 have no risk target. No new quality labels.',
            normalization='Hidden per-input RMS; mean/std floor.1 fit only TRAIN. No development fitting.',
            decoder='Parent keep/refusals exactly preserved. Only parent active tool magnitudes changed, sign/activation preserved. Rounded zero on active tool errors, no fallback. Quality policy vetoes parent edits at fixed risk sigmoid>=.5; this is learned decision output, not calibrated confidence.'),
        evaluation=dict(first='All4 policies/checkpoints on old6; require>=4 both-language clean completions andzero major to advance.',
            native='Native20>=16, decision20exact,zero major; complete native-pixel clarification for any potentially qualified checkpoint before selection.',
            controlled='Only stage1qualifiers may create controlled200 feature cache andbe evaluated; require>=81/100 andzero major.',
            selection='Native clean completions,controlled clean completions,blinded paired review,earlier checkpoint. Never training fit,L1 or parameter error.',
            reuse='Archived R15 parent decisions andR20 hidden inputs only; newhead inference and actual edits are fresh. Historical final exact-pixel AI disputes stay uncertain.',
            acceptance='Closed until route/checkpoint locked andreplicate seed passes all gates. No threshold/parameter tuning after acceptance.'),
        resources=dict(cap_GPU_hours=24,occupied_at_freeze=used,remaining=24-used,allowed_GPU=[0,2,3],GPU1_preserved=True,
            reserve_GPU_hours=.25,initial_training_GPU_hours=0,new_cache_only_after_functional_gate=True,
            future_GPU_jobs_max_seconds=900,shared_watchdog_save_threshold=23.9),
        input_policy='Only current original image,literal request andR15 tool notes enter frozen VLM. No labels,references,review/task/category/IDs. Cached parent JSON is output from that same frozen parent, not reference guidance.',
        code_hashes=hashes,L1_used_for_selection=False,independent_user_feedback=0)
    write(EXP/'training-protocol.json',value)
    write(DATA/'source-manifest.json',dict(at=stamp(),source_round='R20 unchanged reviewed labels',data=value['data'],
        files={name:sha(DATA/name) for name in ('cases.json','train-accepted.json','training-pool.json')},new_visual_labels=0))
    print('Frozen R21 CPU pilot; archived292parent decisions verified; noGPU launched. Remaining',24-used)

if __name__=='__main__':main()

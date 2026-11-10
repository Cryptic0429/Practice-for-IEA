"""Bind the prior R15 baseline to its already verified pretraining archive."""
import datetime as dt
from pathlib import Path
from functional_v1.common import ROOT,read,write,sha,stamp
from .runtime import EXP,PARENT_SHA
from .outcomes import EDITOR_VERSION


def main():
    target=EXP/'baseline-inheritance.json';assert not target.exists()
    folder=ROOT/'archives/functional-v15-20261106'
    receipt=read(folder/'complete-receipt.json')
    snapshot=folder/'complete-snapshot/MANIFEST.json'
    assert sha(snapshot)==receipt['manifest_sha256']=='b002e203c8110a9a87e31525caf73ee24624cb79e919111e6cce10d7f25f9cc0'
    assert receipt['sha256']=='10a2ff7fb4266b81db9aeb1fa5018bb295c3fde5524d7df509ea24c83c8f8e7a'
    manifest={r['path']:r for r in read(snapshot)['files']}
    protocol=read(EXP/'training-protocol.json')
    assert dt.datetime.fromisoformat(receipt['created_at'])<dt.datetime.fromisoformat(protocol['at'])
    evaluation=read(EXP/'evaluation-protocol.json')
    split_keys={'native-development':'native_development','old-native':'old_native_regressions',
        'decision-development':'decision_development','development':'controlled_development'}
    verified={};summaries={}
    for split,key in split_keys.items():
        base=ROOT/'experiments/functional-v15-20261106/evaluation'/EDITOR_VERSION/split/'language-merger-e1'
        names=['predictions.jsonl','identity.json']
        names+=['decision-summary.json'] if split=='decision-development' else ['direct-review.json','direct-functional-summary.json']
        for name in names:
            path=base/name;relative=str(path.relative_to(ROOT))
            assert sha(path)==manifest[relative]['sha256']
            verified[relative]=sha(path)
        identity=read(base/'identity.json')
        assert identity['adapter_sha256']==PARENT_SHA and identity['editor']==EDITOR_VERSION
        assert identity['cases_sha256']==evaluation['case_files'][key]['sha256']
        summary=read(base/('decision-summary.json' if split=='decision-development' else 'direct-functional-summary.json'))
        assert summary['status']=='completed'
        summaries[split]=summary
    parent=dict(native_completed=summaries['native-development']['fully_completed_both_languages'],
        native_major=summaries['native-development']['major_quality_issues'],
        old_completed=summaries['old-native']['fully_completed_both_languages'],old_major=summaries['old-native']['major_quality_issues'],
        decision_controls=summaries['decision-development']['passed_both_languages'],
        controlled_completed=summaries['development']['fully_completed_both_languages'],controlled_major=summaries['development']['major_quality_issues'])
    assert (parent['native_completed'],parent['old_completed'],parent['decision_controls'],parent['controlled_completed'],parent['controlled_major'])==(16,4,20,75,1)
    write(target,dict(at=stamp(),status='exact_pretraining_archive_baseline_verified',parent_adapter_sha256=PARENT_SHA,
        prior_archive=receipt['archive'],prior_archive_sha256=receipt['sha256'],prior_manifest_sha256=receipt['manifest_sha256'],
        archive_snapshot_predates_R18_training_protocol=True,verified_sources=verified,parent_scores=parent,
        new_model_inference_performed=False,new_visual_grades=0,GPU_hours=0,
        limitation='Exact previously AI-reviewed predictions, metadata and scores, with uncertainty retained. Not a new independent benchmark, regrading or user endorsement.'))
    print('Verified frozen R15parent baseline:',parent)


if __name__=='__main__':main()

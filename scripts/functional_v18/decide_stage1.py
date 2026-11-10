"""Keep16/4/20 and zero-major gates; controlled81M0 is still mandatory."""
import argparse
from functional_v1.common import ROOT,read,write,sha,stamp
from .runtime import EXP
from .outcomes import EDITOR_VERSION


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',required=True);parser.add_argument('--name',required=True)
    args=parser.parse_args();assert args.name.replace('-','').isalnum()
    target=EXP/(args.name+'.json');assert not target.exists()
    evaluation=read(EXP/'evaluation-protocol.json');gate=evaluation['stage1']
    inheritance=read(EXP/'baseline-inheritance.json')
    assert inheritance['status']=='exact_pretraining_archive_baseline_verified'
    for path,value in inheritance['verified_sources'].items():assert sha(ROOT/path)==value
    results={}
    parent=inheritance['parent_scores']
    assert (parent['native_completed'],parent['old_completed'],parent['decision_controls'],parent['controlled_completed'],parent['controlled_major'])==(16,4,20,75,1)
    for mid in args.models:
        base=EXP/'evaluation'/EDITOR_VERSION
        native=read(base/'native-development'/mid/'direct-functional-summary.json')
        old=read(base/'old-native'/mid/'direct-functional-summary.json')
        control=read(base/'decision-development'/mid/'decision-summary.json')
        assert native['status']==old['status']=='completed' and not native['missing_unique_images'] and not old['missing_unique_images']
        row=dict(native_completed=native['fully_completed_both_languages'],native_major=native['major_quality_issues'],
            old_completed=old['fully_completed_both_languages'],old_major=old['major_quality_issues'],decision_controls=control['passed_both_languages'],
            native_refusals=native['declined_semantic_cases'],old_refusals=old['declined_semantic_cases'],new_model_inference_performed=True)
        row['stage1_passed']=row['native_completed']>=gate['native_development_minimum'] and row['old_completed']>=gate['old_native_minimum'] and row['native_major']==row['old_major']==0 and row['decision_controls']==gate['decision_controls_required']
        results[mid]=row
    write(target,dict(at=stamp(),parent_baseline=parent,parent_new_inference_performed=False,results=results,
        qualified_checkpoints=[m for m in args.models if results[m]['stage1_passed']],stage2_required_before_selection=True,
        evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),training_protocol_sha256=sha(EXP/'training-protocol.json'),
        baseline_inheritance_sha256=sha(EXP/'baseline-inheritance.json'),L1_used=False,independent_user_feedback=0,final_acceptance_closed=True))
    print({mid:row['stage1_passed'] for mid,row in results.items()})


if __name__=='__main__':main()

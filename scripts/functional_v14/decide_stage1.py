"""Apply the frozen functional gates without relaxing thresholds or using loss."""
import argparse
from functional_v1.common import read,write,sha,stamp
from functional_v11.outcomes import EDITOR_VERSION
from .runtime import EXP


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',required=True);p.add_argument('--name',required=True);a=p.parse_args()
    assert a.name.replace('-','').isalnum()
    dest=EXP/f'{a.name}.json';assert not dest.exists()
    protocol=read(EXP/'evaluation-protocol.json');gate=protocol['stage1'];results={}
    for mid in ['parent-frozen']+a.models:
        base=EXP/'evaluation'/EDITOR_VERSION
        native=read(base/'native-development'/mid/'direct-functional-summary.json')
        old=read(base/'old-native'/mid/'direct-functional-summary.json')
        control=read(base/'decision-development'/mid/'decision-summary.json')
        assert native['status']==old['status']=='completed' and not native['missing_unique_images'] and not old['missing_unique_images']
        r=dict(native_completed=native['fully_completed_both_languages'],native_major=native['major_quality_issues'],
            old_completed=old['fully_completed_both_languages'],old_major=old['major_quality_issues'],
            decision_controls=control['passed_both_languages'],native_refusals=native['declined_semantic_cases'],
            old_refusals=old['declined_semantic_cases'],new_model_inference_performed=mid!='parent-frozen')
        r['stage1_passed']=r['native_completed']>=gate['native_development_minimum'] and r['old_completed']>=gate['old_native_minimum'] and r['native_major']==r['old_major']==0 and r['decision_controls']==gate['decision_controls_required']
        results[mid]=r
    assert results['parent-frozen']['native_completed']==15
    qualified=[mid for mid in a.models if results[mid]['stage1_passed'] and results[mid]['native_completed']>results['parent-frozen']['native_completed']]
    write(dest,dict(at=stamp(),results=results,qualified_checkpoints=qualified,stage2_required_before_selection=True,
        evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),training_protocol_sha256=sha(EXP/'training-protocol.json'),
        L1_used=False,independent_user_feedback=0,final_acceptance_closed=True))
    print('Stage1',results,'qualified',qualified)


if __name__=='__main__':main()

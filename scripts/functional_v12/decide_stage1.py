"""Apply pre-frozen functional gates after complete direct image review."""
from functional_v1.common import ROOT,read,write,sha,stamp
from .runtime import EXP
from .outcomes import EDITOR_VERSION


def main():
    destination=EXP/'stage1-decision.json';assert not destination.exists()
    results={}
    for model in ('parent-new-manual','language-merger-e1'):
        base=EXP/'evaluation'/EDITOR_VERSION
        native=read(base/'native-development'/model/'direct-functional-summary.json')
        old=read(base/'old-native'/model/'direct-functional-summary.json')
        controls=read(base/'decision-development'/model/'decision-summary.json')
        assert native['status']==old['status']=='completed'
        assert not native['missing_unique_images'] and not old['missing_unique_images']
        r=dict(native_completed=native['fully_completed_both_languages'],native_major=native['major_quality_issues'],
            old_completed=old['fully_completed_both_languages'],old_major=old['major_quality_issues'],
            decision_controls=controls['passed_both_languages'],feasible_native_refusals=native['declined_semantic_cases'],
            feasible_old_refusals=old['declined_semantic_cases'],model_inference_performed=True)
        r['stage1_passed']=r['native_completed']>=16 and r['old_completed']>=4 and r['native_major']==r['old_major']==0 and r['decision_controls']==20
        results[model]=r
    candidate=results['language-merger-e1'];control=results['parent-new-manual']
    candidate['not_worse_than_new_manual_control']=candidate['native_completed']>=control['native_completed'] and candidate['old_completed']>=control['old_completed'] and candidate['native_major']<=control['native_major'] and candidate['old_major']<=control['old_major'] and candidate['decision_controls']>=control['decision_controls']
    qualified=['language-merger-e1'] if candidate['stage1_passed'] and candidate['not_worse_than_new_manual_control'] else []
    write(destination,dict(at=stamp(),results=results,qualified_checkpoints=qualified,
        evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),
        training_protocol_sha256=sha(EXP/'training-protocol.json'),
        stage2_required_before_selection=True,L1_used=False,independent_user_feedback=0,
        prompt_control='Exact frozen R11 parent weights, new manual, actual model inference; no new training. Separates prompt and training contribution.'))
    print(results)


if __name__=='__main__':main()

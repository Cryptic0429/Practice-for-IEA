"""Freeze the unsuccessful input experiment and expose diagnostic epoch-one adapters."""
import shutil
from .common import ROOT, DATA, EXP, EVAL, PARENT, read, write, sha, stamp


def main():
    destination = EXP/'development-selection.json'
    if destination.exists():
        raise RuntimeError('Selection already frozen')
    ids = ['parent-plain', 'parent-measured', 'plain-e1', 'plain-e2', 'measured-e1', 'measured-e2']
    scores = {split: {mid: read(EVAL/split/mid/'direct-functional-summary.json') for mid in ids}
              for split in ('development', 'natural-use')}
    assert all(s['status'] == 'completed' for group in scores.values() for s in group.values())
    gate = read(EXP/'protocol.json')['promotion_gate']
    qualified = [mid for mid in ids if not mid.startswith('parent-')
                 and scores['development'][mid]['major_quality_issues'] == gate['major_quality_cases']
                 and scores['natural-use'][mid]['major_quality_issues'] == gate['major_quality_cases']
                 and scores['development'][mid]['fully_completed_both_languages'] >= gate['controlled_qualified_at_least']
                 and scores['natural-use'][mid]['fully_completed_both_languages'] > gate['native_qualified_more_than']]
    assert not qualified, 'Qualified routes require a separate replication/final-lock workflow'
    result = dict(status='input_pilot_failed_practical_goal_not_achieved', frozen_at=stamp(),
                  development_results=scores['development'], natural_results=scores['natural-use'],
                  promoted=False, acceptance_status='closed_not_evaluated', models={},
                  seed_replication_started=False, trial_default_changed=False,
                  reason='No checkpoint passes the predeclared practical gate. Measured inputs do not improve the six original-request probes beyond 3/6, and every trained checkpoint retains severe exposure damage.',
                  diagnostic_policy='No eligible winner exists. Epoch one of each route is exposed only as an early reference, not selected as a qualified champion. All four checkpoint results remain visible.')
    for route in ('plain', 'measured'):
        mid = route+'-e1'
        source = EXP/'runs'/f'{route}-seed-20261108'/'checkpoint-230'
        dest = ROOT/'models/from-functional-v2'/f'sft-input-{route}'/'seed-01'
        dest.mkdir(parents=True, exist_ok=False)
        for filename in ('adapter_config.json', 'adapter_model.safetensors'):
            shutil.copy2(source/filename, dest/filename)
        item = dict(model_id=f'functional-v4-{route}-20261108', evaluation_id=mid,
                    display_name='第四轮 · '+('图像观测输入' if route == 'measured' else '普通输入对照')+'（未通过）',
                    trial_status='第四轮未通过升级门槛；仅供诊断试用，存在过曝和复合要求漏项。',
                    adapter=str(dest.relative_to(ROOT)), adapter_sha256=sha(dest/'adapter_model.safetensors'),
                    parent=str(PARENT.relative_to(ROOT)), parent_sha256=sha(PARENT/'adapter_model.safetensors'),
                    checkpoint=str(source.relative_to(ROOT)), epoch=1, step=230,
                    dataset_sha256=sha(DATA/'train-accepted.json'), input_route=route,
                    runtime_family='functional_v4', runtime_sha256=sha(ROOT/'scripts/functional_v4/runtime.py'),
                    status='diagnostic_trial_only_not_promoted', diagnostic_choice=result['diagnostic_policy'],
                    load_method='Original Qwen base plus this one adapter; do not stack parent adapters.',
                    development_summary=scores['development'][mid], natural_summary=scores['natural-use'][mid],
                    human_feedback=False, frozen_at=stamp())
        write(dest/'metadata.json', item)
        result['models'][item['model_id']] = item
    write(destination, result)
    print(result['status'])


if __name__ == '__main__':
    main()

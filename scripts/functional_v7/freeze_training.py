"""Lock the admitted data, matched runs and functional gates before GPU work."""
import copy
import hashlib
from collections import Counter
from functional_v1.common import ROOT, read, write, sha, stamp
from .diagnose_editor import EXP
from .admit_training import DATA
from .outcomes import system_prompt, EDITOR_VERSION, SCHEMA_VERSION


def main():
    assert not (EXP / 'training-protocol.json').exists() and not (EXP / 'evaluation-protocol.json').exists()
    admission = read(DATA / 'admission.json')
    assert admission['source_groups'] == 160 and admission['photo_domains'] == dict(portrait=80, landscape=80)
    assert sha(DATA / 'cases.json') == admission['cases_sha256']
    assert sha(DATA / 'train-accepted.json') == admission['dataset_sha256']
    transfer = read(EXP / 'runtime-transfer-audit.json')
    assert transfer['runtime_sha256'] == sha(ROOT / 'scripts/functional_v7/runtime.py')
    previous = read(ROOT / 'experiments/functional-v6-20261105/evaluation-protocol.json')
    case_files = copy.deepcopy(previous['case_files'])
    for item in case_files.values():
        assert sha(ROOT / item['path']) == item['sha256']
    native = read(ROOT / case_files['native_development']['path'])
    groups = []
    for domain in ('portrait', 'landscape'):
        unique = {c['group_id']: c for c in native if c['domain'] == domain}
        groups.extend(sorted(unique.values(), key=lambda c: c['group_id'])[:5])
    assert len(groups) == 10 and Counter(c['domain'] for c in groups) == dict(portrait=5, landscape=5)
    controls = []
    for i, raw in enumerate(groups):
        for action in ('keep', 'decline'):
            c = copy.deepcopy(raw)
            if action == 'keep':
                instructions = dict(zh='请保持这张照片完全不变，直接返回原图。', en='Leave this photograph completely unchanged and return the original.')
            else:
                pairs = [
                    ('在照片右下角添加一只白色小狗。', 'Add a small white dog in the lower-right corner of the photograph.'),
                    ('在画面上方添加文字“旅行日记”。', 'Add the text "Travel Diary" across the top of the photograph.'),
                    ('把照片中间的一块区域变成透明背景，其他地方不改。', 'Make a region in the centre of the photograph transparent while leaving other regions unchanged.'),
                    ('在照片中央添加一个红色气球。', 'Add a red balloon at the centre of the photograph.')]
                zh, en = pairs[i % len(pairs)]
                instructions = dict(zh=zh, en=en)
            c.update(id='decision-dev-' + raw['id'] + '-' + action, instructions=instructions,
                     family='decision_control', expected_action=action,
                     expected_reason='unsupported_tools' if action == 'decline' else None,
                     action='decision_control', instruction_origin='AI-authored developer control; unchanged original development photo')
            controls.append(c)
    path = EXP / 'evaluation/decision-development-cases.json'
    write(path, controls)
    case_files['decision_development'] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path))
    write(EXP / 'evaluation-protocol.json', dict(
        at=stamp(), status='functional_and_decision_gates_frozen_before_round7_training_or_new_model_predictions',
        case_files=case_files, unit='Semantic cases grouped by original photograph; both ZH and EN must complete with acceptable quality. Translations are not independent cases.',
        stage1=dict(native_development_minimum=16, native_development_total=20, old_native_minimum=4, old_native_total=6,
                    maximum_major_cases=0, decision_controls_total=20, decision_controls_required=20,
                    decisions='Ten unsupported requests must decline with unsupported_tools and no result; ten explicit no-ops must keep the exact input pixels. Both languages required. Controls are on ten already reused developer photos, not an independent dataset.'),
        stage2=dict(controlled_development_minimum=81, controlled_development_total=100, maximum_major_cases=0,
                    run_only_after_stage1_passes=True),
        assessment='Codex direct executed-image AI review with model identity hidden during grading; all uncertainty and same-pixel conflicts preserved. Frozen raw7B visual screening previously failed directional calibration and is reference-only, not an independent reliable judge.',
        refusal_policy='On feasible native/controlled tasks, every decline remains uncompleted and is separately reported as a refusal. No result image or input fallback is allowed for decline. Fewer outputs cannot inflate completion counts.',
        comparisons=['Current default and round6 parent under the same new prompt/schema/editor',
                     'Both matched new training routes under that exact runtime',
                     'Previously frozen current-default and round6-parent original-pipeline outcomes',
                     'Recorded old parameter predictions replayed through the new editor to isolate editor contribution; these are not new-prompt inference'],
        selection=dict(eligibility='Pass all functional/decision gates, exceed the original-pipeline functional default and add no major damage.',
                       rank=['Most completed acceptable semantic tasks', 'Fewer major effects', 'Anonymous paired visual comparison', 'Earlier checkpoint'],
                       L1_or_parameter_error_used=False),
        acceptance=previous['acceptance'], replicate_only_after_all_development_gates=True,
        preservation='Existing native20 and controlled100 acceptance files unchanged and remain sealed for inference. Never choose a checkpoint or modify parameters after acceptance.',
        resource=previous['resource'], new_training_started=False, actual_user_feedback=0))
    rows = admission['actual_rows_each_route']
    assert rows % 4 == 0
    settings = dict(rows=rows, epochs=4, steps_per_epoch=rows // 4, steps=rows,
                    learning_rate=5e-5, effective_batch=4, rank=8, alpha=16, dropout=0.,
                    seed=20261112, replicate_seed=20261113, per_device_batch=1, gradient_accumulation=4,
                    image_max_pixels=262144, max_sequence_tokens=2048, quantization='NF4, double quantization, BF16 compute',
                    optimizer='adamw_torch', weight_decay=0., scheduler='cosine', warmup_ratio=.05,
                    max_grad_norm=1., save='every complete epoch', assistant_only_decision_JSON_supervision=True)
    files = ['scripts/functional_v1/common.py', 'scripts/functional_v1/editor.py', 'scripts/functional_v1/runtime.py',
             'scripts/functional_v7/editor_soft.py', 'scripts/functional_v7/outcomes.py', 'scripts/functional_v7/runtime.py',
             'scripts/functional_v7/train.py', 'scripts/functional_v7/evaluate.py', 'scripts/functional_v7/freeze_training.py']
    parent = ROOT / 'models/from-functional-v2/sft-native-coverage/seed-01'
    assert sha(parent / 'adapter_model.safetensors') == '39d0e3b91c321ee0abeca8f3260116c76405d371127ba118e04822701e09f66c'
    write(EXP / 'training-protocol.json', dict(
        at=stamp(), status='actual_data_runtime_optimizer_and_functional_gates_frozen_before_training',
        hypothesis='After explicit outcome-schema supervision and quality-audited tool targets, test whether adapting the visual merger improves image-conditioned editing over language-only LoRA. Both routes inherit the same112 exact parent tensors; the expanded route adds four merger tensors with zero-initialized B matrices, never stacked adapters.',
        routes=['language', 'language_merger'], parent=str(parent.relative_to(ROOT)), parent_sha256=sha(parent / 'adapter_model.safetensors'),
        training=settings, dataset_sha256=sha(DATA / 'train-accepted.json'), cases_sha256=sha(DATA / 'cases.json'),
        admission_sha256=sha(DATA / 'admission.json'), input_audit_sha256=sha(EXP / 'input-audit.json'),
        target_replay_sha256=sha(EXP / 'target-replay.json'), evaluation_protocol_sha256=sha(EXP / 'evaluation-protocol.json'),
        CPU_transfer_audit_sha256=sha(EXP / 'runtime-transfer-audit.json'),
        code_hashes={name: sha(ROOT / name) for name in files},
        editor=EDITOR_VERSION, schema=SCHEMA_VERSION, runtime='functional_v7.runtime',
        system_prompt_sha256=hashlib.sha256(system_prompt().encode()).hexdigest(),
        source_counts=admission, resource='Original shared16 occupied GPU hours; GPU0/2/3 only, GPU1 preserved. Reserve budget for evaluation and conditional replication.'))
    print('Locked', admission['semantic_cases'], 'semantic cases,', rows, 'bilingual rows; both matched four-epoch routes; no GPU work yet.')


if __name__ == '__main__':
    main()

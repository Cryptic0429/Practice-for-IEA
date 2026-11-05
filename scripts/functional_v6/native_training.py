"""Prepare a bilingual, source-audited native cohort and collect proposals, never auto-admit."""
import argparse
import json
import os
from collections import Counter
from functional_v1.common import ROOT, read, write, append, sha, stamp, GPUJob, system_prompt
from .source_bank import EXP, DATA
from .isolation import observed_bank


# Chosen from source/request observations before any proposal output. Prefix n = initial bank,
# x = extra original training pool. Chinese translations do not replace original English.
SPECS = [
('n001','brown_tone','让照片更有棕褐色风格。'),
('n011','compound','请提亮照片，并减弱其中的棕色调。'),
('n021','brown_tone','添加棕色调。'),
('n022','brown_tone','请给照片添加棕色、较暖的色调。'),
('n026','brown_tone','请添加少量棕色调。'),
('n031','brown_tone','请给照片添加少量棕色调。'),
('n036','brown_tone','把照片色调改成棕褐色。'),
('n041','white_balance','请去除照片中的棕色调。'),
('n071','compound','重新调整色调，让照片更偏棕褐色，并稍微提亮。'),
('n164','brown_tone','请给照片添加少量棕色调。'),
('n012','compound','添加冷色调，提高对比度。'),
('n017','compound','提亮照片，并添加暖色调。'),
('n027','white_balance','给照片添加暖色调。'),
('n032','compound','增强色彩，并让色调更冷一些。'),
('n037','white_balance','让颜色稍微暖一点。'),
('n057','compound','提高亮度，并调暖一些。'),
('n062','white_balance','让整张照片的色调更暖。'),
('n165','compound','提高对比度，并减弱黄色偏色。'),
('n097','white_balance','让照片色调更暖。'),
('n142','compound','让照片更暖、饱和度更高。'),
('n003','compound','提高对比度，让颜色更鲜明。'),
('n013','compound','请大幅提高对比度，并增强色彩。'),
('n018','contrast','提高对比度。'),
('n028','contrast','提高对比度。'),
('n043','contrast','给照片增加一些对比度。'),
('n048','compound','提高对比度，并提亮照片。'),
('n058','compound','稍微调暗一点，增强色彩并提高对比度。'),
('n063','compound','稍微提高亮度和对比度。'),
('n068','compound','大幅提高整张照片的对比度，并增加琥珀色或玫瑰金色调。'),
('n161','contrast','提高对比度。'),
('n064','saturation','提高照片饱和度。'),
('n087','compound','让颜色更暖、更浓郁。'),
('n150','saturation','让照片色彩更丰富、更鲜明。'),
('n179','saturation','降低鲜艳程度，让色彩更柔和、更自然。'),
('n185','saturation','让颜色不那么鲜艳。'),
('n188','saturation','增强色彩，让颜色更鲜明。'),
('n191','saturation','让颜色不那么鲜艳。'),
('n206','saturation','增强色彩，让颜色更鲜明、更有表现力。'),
('n209','saturation','增强色彩，让颜色更鲜明、更有表现力。'),
('n238','contrast','稍微提高色彩对比度。'),
('n005','lighting','提亮整张照片。'),
('n010','lighting','提亮照片。'),
('n015','lighting','提高曝光。'),
('n019','compound','把照片调暗，并让颜色更丰富。'),
('n020','lighting','大幅提亮照片。'),
('n025','lighting','请提亮照片。'),
('n070','lighting','请大幅提亮整张照片。'),
('n127','lighting','能把这张照片大幅提亮吗？'),
('n143','lighting','把照片稍微提亮一点。'),
('n204','lighting','大幅提高亮度。'),
('n004','saturation','让照片色彩更鲜明。'),
('n024','compound','略微提高亮度，也略微提高饱和度。'),
('n029','compound','纠正过曝，并增强色彩。'),
('n030','lighting','请提亮整张照片。'),
('n039','saturation','提高饱和度。'),
('n042','compound','提亮照片，调整白平衡，让色调更冷。'),
('n054','saturation','我想让这张照片色彩更丰富、更有吸引力。'),
('n066','brown_tone','让这张照片有棕褐色效果。'),
('n081','brown_tone','添加淡淡的棕褐色调。'),
('n083','contrast','提高对比度。'),
('n089','saturation','增强色彩，让颜色更鲜明、更有表现力。'),
('n093','compound','提高对比度和亮度，让色彩更丰富。'),
('n101','brown_tone','请给照片添加棕色调。'),
('n107','compound','请把照片稍微提亮一点，并让颜色更暖。'),
('n119','lighting','提亮整张照片。'),
('n180','compound','请提亮照片，让颜色更鲜明。'),
('n186','lighting','提亮照片，使建筑物足够清楚。'),
('n189','lighting','把照片提亮一点。'),
('n190','compound','请大幅提高对比度，并明显增强色彩。'),
('n194','saturation','增强色彩，让颜色更鲜明。'),
('n215','saturation','提高照片饱和度。'),
('n224','saturation','请略微提高饱和度。'),
('n225','lighting','请降低亮度。'),
('n227','saturation','增强色彩，让颜色更鲜明、更有表现力。'),
('n232','compound','略微提高对比度和亮度。'),
('n235','contrast','能增强这张照片明暗之间的对比吗？'),
('x020','saturation','请让颜色更鲜明。'),
('x030','lighting','请把照片大幅提亮。'),
('x035','compound','请降低亮度，并让颜色不那么鲜艳。'),
('x038','saturation','略微提高饱和度。'),
('x053','compound','让照片色彩更丰富、更明亮。'),
('x071','compound','提亮照片，让颜色更鲜明。'),
('x092','compound','请降低亮度，并增强照片色彩。'),
('x113','saturation','让颜色更鲜明，并略微提高饱和度。'),
('x121','compound','大幅提高亮度和对比度。'),
('x137','saturation','增强色彩，让颜色更鲜明。'),
('x151','compound','稍微提亮，改善光照，并略微降低对比度。'),
('x160','compound','略微提高对比度，并突出蓝色调。'),
('x161','saturation','增强色彩，让颜色更鲜明、更有表现力。'),
('x186','lighting','请提亮照片。'),
('x191','compound','提亮照片，并提高饱和度。'),
('x194','compound','请提亮照片，让颜色更鲜明。'),
('x196','contrast','请给照片增加对比度。'),
('x202','compound','提高饱和度和对比度。'),
('x205','compound','添加蓝色调，略微提高亮度和对比度。'),
('x206','saturation','增强色彩，让颜色更鲜明、更有表现力。'),
('x231','lighting','调暗整张照片。'),
('x246','lighting','请提亮照片。'),
('x250','compound','提亮照片，提高对比度。'),
('x351','lighting','大幅提亮整张照片。'),
]


def prepare():
    from functional_v1.editor import load_rgb, save
    dest = DATA/'native-training-candidates.json'
    assert not dest.exists(), 'Source cohort/requests already prepared'
    bank, notes, _ = observed_bank()
    audit_path = EXP/'planning/observed-source-isolation.json'; audit = read(audit_path)
    for path, value in audit['source_files'].items(): assert sha(ROOT/path) == value
    accept_path = DATA/'acceptance-native-cases.json'
    assert sha(accept_path) == read(DATA/'acceptance-native-freeze.json')['cases_sha256']
    cases = []
    for short, family, zh in SPECS:
        sid = ('train-native-' if short[0] == 'n' else 'train-extension-')+short[1:]
        source, note = bank[sid], notes[sid]
        assert sid != 'train-native-168', 'User-shown case12 remains disclosed regression, not new training'
        assert source['split'] == 'train' and note['source_usable'] and note['request_scope'] == 'global'
        assert source['source_sha256'] == sha(source['source_image'])
        image = DATA/'inputs'/f'native-train-{short}.png'
        save(image, load_rgb(source['source_image']))
        cases.append(dict(id=f'native-train-{short}', source_id=sid, group_id=source['group_id'],
            domain=note['domain'], split='train', family=family, action='original-native-request',
            image=str(image.relative_to(ROOT)), input_sha256=sha(image), instructions={'en': source['instruction'], 'zh': zh},
            expected_change=zh+' 请求涉及多项时，每项都必须在图片上完成；力度以原始措辞及当前图像状态判断。',
            must_preserve='主体、构图、未要求的色相和主要明暗细节；保留自然肤色、天空及浅色衣物，不新增明显过曝、压黑、失真或无关效果。已有缺失细节不凭空恢复。',
            failure_conditions=['方向错误或无可见效果', '复合要求漏项', '力度不足或过强', '重大过曝、压黑、偏色或额外编辑'],
            instruction_origin='English untouched original GIER annotation; Chinese AI translation; no AI rewrite or synthetic reversal of the English request.',
            source_partition='original training_pool', source_record_id=source['record_id'],
            original_source_sha256=source['source_sha256'], native_annotation=source['native_annotation'],
            source_classification_evidence=note['evidence'], target_status='not_yet_proposed_or_visually_admitted'))
    assert len(cases) == len({c['group_id'] for c in cases}) == 100
    assert Counter(c['domain'] for c in cases) == {'portrait': 50, 'landscape': 50}
    selected = {c['source_id'] for c in cases}
    for c in audit['collisions']:
        if c['a'] in selected: raise AssertionError(c)
        if c['reason'] == 'near_current_sources' and c['b'] in selected: raise AssertionError(c)
    old_train = {c['group_id'] for c in read(ROOT/'data/training/functional-v4-20261105/cases.json')}
    held = {c['group_id'] for c in read(accept_path)+read(DATA/'development-native-cases.json')}
    assert not {c['group_id'] for c in cases} & (old_train | held)
    write(dest, cases)
    from functional_v3.common import PARENT
    write(EXP/'native-proposal-protocol.json', dict(at=stamp(), status='source_cohort_only_not_admitted_training_dataset',
        native_cases_sha256=sha(dest), native_photos=100, domains={'portrait': 50, 'landscape': 50}, language_rows=200,
        accepted_targets=0, family_counts=dict(Counter(c['family'] for c in cases)),
        policy='100 different original training photos, original English requests plus Chinese AI translations. Source observation and isolation only; no target is approved until actual executed output is directly reviewed. Failed labels may be revised or replaced with an explicitly recorded training-source amendment; no heldout can be replaced by model outcomes.',
        proposal_models={'base': None, 'parent': str(PARENT.relative_to(ROOT))}, parent_sha256=sha(PARENT/'adapter_model.safetensors'),
        system_prompt_sha256=__import__('hashlib').sha256(system_prompt().encode()).hexdigest(),
        code_sha256={f: sha(ROOT/f) for f in ('scripts/functional_v1/common.py', 'scripts/functional_v1/runtime.py', 'scripts/functional_v1/editor.py', 'scripts/functional_v6/native_training.py')},
        input_policy='Only original image, original request and exact v1 tool prompt; no reference, old label, quality note, image statistic string or target metadata enters inference.',
        acceptance_cases_sha256=sha(accept_path), development_cases_sha256=sha(DATA/'development-native-cases.json'),
        source_isolation_sha256=sha(audit_path), user_case12_excluded_from_new_training=True,
        planned_controlled_tasks=360, planned_controlled_photos=60,
        prospective_combination='Only if all100 native targets pass visual admission:460 semantics920 distinct bilingual rows160 groups80portrait/80landscape; no native oversampling. This is a plan, not an actual admitted or trained count.',
        no_round6_model_training_started=True))
    print('Native source cohort prepared:100photos50/50,200bilingual requests,0labels admitted')


def collect(route, gpu):
    from functional_v3.common import PARENT
    protocol = read(EXP/'native-proposal-protocol.json')
    assert sha(DATA/'native-training-candidates.json') == protocol['native_cases_sha256']
    for path, value in protocol['code_sha256'].items(): assert sha(ROOT/path) == value
    assert sha(PARENT/'adapter_model.safetensors') == protocol['parent_sha256']
    assert route in ('base', 'parent')
    cases = read(DATA/'native-training-candidates.json')
    out = EXP/'native-proposals'/route; path = out/'predictions.jsonl'
    rows = list(map(json.loads, path.read_text().splitlines())) if path.exists() else []
    done = {r['case_id'] for r in rows}
    assert len(done) == len(rows)
    if len(done) == len(cases): return
    with GPUJob(gpu, 'v6-native-training-proposals-'+route, max_seconds=5400) as budget:
        import torch
        from functional_v1.runtime import load_model, processor, predict
        from functional_v1.editor import load_rgb, edit, save
        torch.set_num_threads(4)
        model = load_model(None if route == 'base' else PARENT); p = processor()
        for case in cases:
            if case['id'] in done: continue
            budget.tick()
            assert case['split'] == 'train' and sha(ROOT/case['image']) == case['input_sha256']
            row = dict(case_id=case['id'], route=route, language='en', input_sha256=case['input_sha256'],
                instruction=case['instructions']['en'], status='failed', role='training_label_proposal_not_model_evaluation_or_label_admission')
            try:
                response, parameters = predict(model, p, case, 'en')
                output = out/'images'/f'{case["id"]}.png'
                save(output, edit(load_rgb(ROOT/case['image']), parameters))
                row.update(status='executed', response=response, parameters=parameters,
                    output_image=str(output.relative_to(ROOT)), output_sha256=sha(output))
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
            row['at'] = stamp(); append(path, row); rows.append(row); done.add(case['id'])
            write(out/'status.json', dict(at=stamp(), pid=os.getpid(), status='running', total=len(cases), completed=len(done),
                executed=sum(r['status'] == 'executed' for r in rows), labels_admitted=0))
        write(out/'status.json', dict(at=stamp(), status='completed', total=len(cases), completed=len(done),
            executed=sum(r['status'] == 'executed' for r in rows), failed=sum(r['status'] != 'executed' for r in rows), labels_admitted=0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['prepare', 'collect'])
    ap.add_argument('--route', choices=['base', 'parent']); ap.add_argument('--gpu', choices=['0', '2', '3'])
    args = ap.parse_args()
    if args.command == 'prepare': prepare()
    else: collect(args.route, args.gpu)

"""Training-only edit/decline candidates, with source and failed-edit provenance."""
import copy
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.editor import load_rgb, save
from functional_v1.curate import font
from .diagnose_editor import EXP
from .outcomes import validate_decision
from .editor_soft import edit


def cards(path, records):
    canvas = Image.new('RGB', (1800, 1500), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), 'TRAINING CANDIDATES | expected decline: unsupported tools | no result image', fill='black', font=font(23))
    for i, c in enumerate(records):
        x, y = (i % 2) * 900, 60 + (i // 2) * 710
        draw.text((x + 12, y), c['id'], fill='black', font=font(21))
        for j, line in enumerate(textwrap.wrap(c['instructions']['en'], 78)):
            draw.text((x + 12, y + 30 + j * 24), line, fill='black', font=font(18))
        with Image.open(ROOT / c['image']) as source:
            im = ImageOps.contain(source.convert('RGB'), (870, 570), Image.Resampling.LANCZOS)
        canvas.paste(im, (x + 12 + (870 - im.width) // 2, y + 115 + (570 - im.height) // 2))
    canvas.save(path, quality=95, subsampling=0)


def pair(path, c, second, second_title):
    canvas = Image.new('RGB', (1800, 800), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), c['id'], fill='black', font=font(22))
    for j, line in enumerate(textwrap.wrap(c['instructions']['en'], 140)):
        draw.text((12, 44 + j * 26), line, fill='black', font=font(19))
    for i, (title, image) in enumerate([('INPUT', c['image']), (second_title, second)]):
        draw.text((i * 900 + 12, 110), title, fill='black', font=font(19))
        with Image.open(ROOT / image) as source:
            im = ImageOps.contain(source.convert('RGB'), (870, 630), Image.Resampling.LANCZOS)
        canvas.paste(im, (i * 900 + 12 + (870 - im.width) // 2, 150 + (630 - im.height) // 2))
    canvas.save(path, quality=95, subsampling=0)


def main():
    out = EXP / 'decision-candidates'
    assert not (out / 'protocol.json').exists()
    source = ROOT / 'data/training/functional-v6-20261105/cases.json'
    old = read(source)
    by_id = {c['id']: c for c in old}
    unsupported = [
        ('train-002', '请移除照片中的人物，并自然补全背景。', 'Remove the people in the photograph and naturally fill the background.'),
        ('train-007', '请将人物身后的背景换成雪山。', 'Replace the background behind the person with snowy mountains.'),
        ('train-010', '只把人物的衣服换成红色，其他区域保持原样。', 'Change only the person\'s clothing to red; leave all other regions unchanged.'),
        ('train-013', '把人物抠出来，导出透明背景图片。', 'Cut out the person and export the photograph with a transparent background.'),
        ('train-016', '请移除人物，保留并补全原来的环境。', 'Remove the person while retaining and filling the original surroundings.'),
        ('train-020', '在人物身旁添加一只白色小狗。', 'Add a small white dog beside the person.'),
        ('train-025', '只修掉脸上的小瑕疵，头发和背景不要改。', 'Retouch small facial blemishes only; leave the hair and background unchanged.'),
        ('train-031', '把人物背景换成室内摄影棚。', 'Replace the background behind the person with an indoor photography studio.'),
        ('train-034', '只把人物的帽子改成红色，其他颜色不变。', 'Change only the person\'s hat to red without changing other colours.'),
        ('train-041', '请移除人物并自然修补被遮挡的背景。', 'Remove the person and naturally reconstruct the background hidden behind them.'),
        ('train-021', '在画面前景添加一只红色行李箱。', 'Add a red suitcase to the foreground.'),
        ('train-027', '请把天空替换成有晚霞的天空，地面完全不改。', 'Replace the sky with a sunset sky while leaving the ground completely unchanged.'),
        ('train-039', '只修改照片上半部分的颜色，其他像素保持原样。', 'Change colours only in the upper half of the photo; leave all other pixels unchanged.'),
        ('train-043', '移除照片中的房屋，并自然补全周围场景。', 'Remove the house and naturally fill the surrounding scene.'),
        ('train-074', '只把屋顶换成鲜红色，其他颜色都不变。', 'Change only the roof to bright red; leave all other colours unchanged.'),
        ('train-086', '在前景添加一个戴黄色帽子的行人。', 'Add a person wearing a yellow hat in the foreground.'),
        ('train-100', '在画面右下角添加一只黑猫。', 'Add a black cat in the lower-right corner of the photograph.'),
        ('train-114', '移除小船和船上的人，并补全水面。', 'Remove the small boat and its passenger, filling the water surface naturally.'),
        ('train-120', '在照片中央添加一棵开花的树。', 'Add a flowering tree at the centre of the photograph.'),
        ('train-149', '在画面上方添加文字“旅行日记”。', 'Add the text "Travel Diary" across the top of the photograph.'),
    ]
    positives = [
        ('native-train-n013', 'brown', {'sepia': 40}, '请给整张照片加上棕色复古色调。', 'Give the whole photograph a brown vintage tone.'),
        ('native-train-n028', 'soft-contrast', {'contrast': -20}, '请降低一些对比度，让明暗过渡柔和一些。', 'Reduce contrast a little to make the tonal transitions softer.'),
        ('native-train-n093', 'monochrome', {'saturation': -100}, '请把照片转换为黑白风格。', 'Convert the photograph to black and white.'),
        ('native-train-n070', 'mild-light', {'exposure': 20}, '请把整张照片稍微提亮一点。', 'Slightly brighten the whole photograph.'),
        ('native-train-n127', 'mild-light', {'exposure': 20}, '请把整张照片稍微提亮一点。', 'Slightly brighten the whole photograph.'),
        ('native-train-n190', 'muted-colour', {'saturation': -25}, '请把颜色的鲜艳程度降低一些。', 'Make the colours a little less vivid.'),
    ]
    quality_ids = ['native-train-n070', 'native-train-n127', 'native-train-n190']
    records = []
    for source_id, zh, en in unsupported:
        c = copy.deepcopy(next(c for c in old if c['source_id'] == source_id))
        c.update(id='unsupported-' + source_id + '-v7', image=c['canonical_photo'],
                 instructions=dict(zh=zh, en=en), family='unsupported', action='decline',
                 instruction_origin='AI-authored unsupported-operation request; original photograph unchanged',
                 english_origin='AI-authored', original_pool_request=None,
                 decision=dict(action='decline', reason='unsupported_tools'), parameters=None,
                 target=None, target_sha256=None, kind='unsupported_decision_candidate',
                 expected_change='Explain unavailable operation and return no image.',
                 must_preserve='No substitute global edit, no edited image or unchanged-image success.',
                 failure_conditions=['Edited image returned for an unavailable operation.', 'Unchanged input reported as an edit.'])
        c['input_sha256'] = sha(ROOT / c['image'])
        records.append(c)
    for cid, suffix, parameters, zh, en in positives:
        c = copy.deepcopy(by_id[cid])
        c.update(id=cid + '-AI-' + suffix + '-v7', instructions=dict(zh=zh, en=en),
                 instruction_origin='AI-authored counterfactual request on unchanged native training photograph; original failed request remains recorded separately',
                 english_origin='AI-authored', original_pool_request=None, parameters=parameters,
                 action='edit', decision=dict(action='edit', parameters=parameters),
                 kind='AI_feasible_counterrequest_candidate',
                 expected_change=en, must_preserve='Original content and basic image quality.',
                 failure_conditions=['Requested direction is absent or opposite.', 'Major added image damage.'])
        c['family'] = {'brown': 'brown_tone', 'soft-contrast': 'contrast', 'monochrome': 'saturation',
                       'mild-light': 'lighting', 'muted-colour': 'saturation'}[suffix]
        records.append(c)
    for cid in quality_ids:
        c = copy.deepcopy(by_id[cid])
        c.update(action='decline', decision=dict(action='decline', reason='unreliable_quality'),
                 parameters=None, target=None, target_sha256=None, kind='quality_decline_candidate',
                 decision_label_origin='Codex AI assessment after recorded original-input parameter replays and full-size quality review; not a human judgment or proof of universal impossibility',
                 expected_change='State that this request cannot be completed reliably with the available tools and return no result.',
                 failure_conditions=['Unreliable edited result returned.', 'Input fallback reported as editing success.'])
        records.append(c)
    c = copy.deepcopy(by_id['native-train-n186'])
    c.update(parameters=dict(exposure=20, shadows=55), decision=dict(action='edit', parameters=dict(exposure=20, shadows=55)),
             kind='same_input_highlight_artifact_repair', action='edit')
    records.append(c)
    assert len(records) == 30 and len({c['id'] for c in records}) == 30
    for c in records:
        assert c['split'] == 'train' and sha(ROOT / c['image']) == c['input_sha256']
        validate_decision(c['decision'])
        c['status'] = 'candidate_pending_direct_visual_and_request_review'
    write(out / 'protocol.json', dict(
        at=stamp(), status='frozen_before_decision_candidate_outputs', source_cases_sha256=sha(source),
        code_sha256=sha(__file__), editor_sha256=sha(Path(__file__).with_name('editor_soft.py')),
        outcome_schema_sha256=sha(Path(__file__).with_name('outcomes.py')),
        source_quality_followup_sha256=sha(EXP / 'label-quality-repairs/full-resolution-observations.json'),
        candidates=records, source_scope='Existing 160 round6 training groups only; original inputs unchanged. No development or acceptance data.',
        disclosure='AI counterrequests are new tasks, not repaired or reclassified original requests. Quality declines reflect current tool reliability, not universal impossibility. Declines remain uncompleted original edits in functional evaluation.',
        admitted=False, GPU_hours=0, new_training_started=False, live_changed=False))
    sheets = []
    for start in range(0, 20, 4):
        path = out / f'unsupported-{start // 4 + 1:02d}.jpg'
        path.parent.mkdir(parents=True, exist_ok=True)
        subset = records[start:start + 4]
        cards(path, subset)
        sheets.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path), case_ids=[c['id'] for c in subset]))
    previous = {
        'native-train-n070': EXP / 'label-quality-repairs/images/native-train-n070-986201080ed5.png',
        'native-train-n127': EXP / 'label-quality-repairs/images/native-train-n127-eeb35d49335c.png',
        'native-train-n190': EXP / 'label-quality-repairs/images/native-train-n190-653f1f055774.png'}
    for c in records[20:]:
        if c['action'] == 'edit':
            result = edit(load_rgb(ROOT / c['image']), c['parameters'], 'bounded-fixed')
            target = out / 'images' / (c['id'] + '.png')
            save(target, result)
            c.update(target=str(target.relative_to(ROOT)), target_sha256=sha(target))
            second, title = c['target'], 'PROPOSED RESULT | not admitted'
        else:
            c['diagnostic_rejected_image'] = str(previous[c['id']].relative_to(ROOT))
            second, title = c['diagnostic_rejected_image'], 'REJECTED DIAGNOSTIC EDIT | expected decline, no result'
        path = out / (c['id'] + '.jpg')
        pair(path, c, second, title)
        sheets.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path), case_ids=[c['id']]))
    write(out / 'manifest.json', dict(at=stamp(), protocol_sha256=sha(out / 'protocol.json'), records=records,
                                      sheets=sheets, admitted=False, new_training_started=False))
    print('Generated 20 unsupported decisions, 6 AI counterrequest edits, 3 quality-decline candidates, 1 natural-request repair. No admission.')


if __name__ == '__main__':
    main()

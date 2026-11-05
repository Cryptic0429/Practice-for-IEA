"""Freeze additional native acceptance requests before any round6 candidate inference."""
from collections import Counter
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.editor import load_rgb, save
from .source_bank import EXP, DATA
from .isolation import observed_bank


SPECS = {
    4: ('lighting', '适度调暗这张照片。', '房屋、积雪和树木整体适度变暗，保留门廊装饰、台阶和雪地层次，不增加色偏。'),
    5: ('compound', '去除棕褐色调，提高对比度，让颜色更自然。', '减弱棕黄偏色，增强明暗分离及自然色彩；不能仅提亮或添加另一种明显偏色，不要求凭空恢复原始未知色彩。'),
    9: ('compound', '提亮阴影，并把整体颜色调得更偏棕褐色。', '暗部面孔更易辨认，同时出现整体棕褐色效果，浅色衣服仍有层次，不以整体过曝替代提暗部。'),
    13: ('brown_tone', '给整张照片添加少量棕色调。', '整体轻微但可辨认地偏棕褐，保留白衬衣和自然人像细节，不能明显过度染黄或泛绿。'),
    14: ('compound', '请提亮照片，调暖并增强色彩，同时提高对比度。', '提亮、暖色、增艳和增对比均可见，面孔、白衬衣与浅色墙仍自然，不能漏项或明显过调。'),
    22: ('compound', '让颜色更鲜明、更明亮。', '海滩及天空反射的颜色更鲜明并有可见的明亮改善，保留云层、水面和小动物结构。'),
    23: ('lighting', '提亮这张照片。', '夜间城市和暗公园整体更清楚，灯光和窗户仍保持主要层次，不能把天空染成不自然的亮色。'),
    25: ('saturation', '大幅降低饱和度。', '黄色衣服和背景颜色明显减淡，仍保留主体与明暗；已有过曝无需虚构恢复，不应进一步扩大过曝。'),
    31: ('saturation', '增强色彩，让颜色更鲜明。', '人物衣物和环境可见增艳，皮肤自然，不额外改变亮暗或色相。'),
    48: ('compound', '大幅提高对比度和饱和度。', '明暗分离与颜色强度均明显提高，保留两个人、衣服及山林层次，不能只改一项或严重失真。'),
    51: ('contrast', '大幅提高对比度。', '温室中明暗分离明显增强，保持屋顶、植物和地面纹理，不能只增亮或额外偏色。'),
    64: ('compound', '提高亮度，改善光照，让颜色更鲜明。', '暗海滩、水面和天空可见提亮并增艳，保留海浪、倒影和远处人物，不变成过亮的白昼画面。'),
    70: ('compound', '请提高亮度，并让照片有更丰富的颜色。', '海浪和夜空适度可见提亮及增艳，保留浪花和暗夜氛围，不严重放大噪声或偏成荧光色。'),
    73: ('saturation', '提高照片的色彩饱和度。', '草地、山体和天空可见增艳，保留遗迹石纹与云层，不产生荧光绿或明显偏色。'),
    77: ('lighting', '把照片稍微调暗一点。', '人物、红色衣服和木门仅温和收暗，保留脸部、白裤和衣物细节，不反向变亮或明显压黑。'),
    84: ('compound', '减弱黄色偏色，并略微提高对比度。', '黄色偏色可见减弱，明暗分离略增，红裙、木门与肤色仍自然，不能转成明显蓝青偏色。'),
    91: ('compound', '提高整张照片的亮度和饱和度。', '星空和银河结构可见提亮并增艳，保留星点与暗天层次，噪声不能压过星空主体。'),
    99: ('compound', '提亮并改善这张照片。', '山谷、村庄整体明亮改善，保留雪山和浅色天空，不以额外强烈色偏代替提亮；没有额外指定的改善不作为必须调用的工具。'),
    101: ('lighting', '提亮这张照片。', '雪地和树林整体可见提亮，保留树干、积雪和远处小人物，不能让积雪大面积发白。'),
    118: ('lighting', '大幅提高亮度。', '两人和室内场景明显提亮，肤色和浅衬衣保持自然层次，不能只轻微改变或大面积过曝。'),
}


def main():
    dest = DATA/'acceptance-native-cases.json'
    assert not dest.exists(), 'Acceptance source/tasks already frozen; never replace for model outcomes'
    bank, notes, _ = observed_bank()
    selected = {f'acceptance-extension-{i:03d}' for i in SPECS}
    audit_path = EXP/'planning/observed-source-isolation.json'
    audit = read(audit_path)
    for path, value in audit['source_files'].items(): assert sha(ROOT/path) == value
    # Every final acceptance source must be free of historical-training and all currently observed-source flags.
    # If a training candidate conflicts with one of these chosen heldouts, exclude training and document it first.
    exclusions = read(EXP/'planning/source-conflict-resolution.json') if (EXP/'planning/source-conflict-resolution.json').exists() else {'excluded_sources': []}
    excluded = set(exclusions['excluded_sources'])
    for c in audit['collisions']:
        if c['reason'] == 'holdout_near_historical_training': assert c['a'] not in selected, c
        if c['reason'] == 'near_current_sources' and (c['a'] in selected or c['b'] in selected):
            other = c['b'] if c['a'] in selected else c['a']
            assert other in excluded and other not in selected, c
    cases = []
    for i, (family, zh, expected) in SPECS.items():
        sid = f'acceptance-extension-{i:03d}'; source, note = bank[sid], notes[sid]
        assert source['split'] == 'acceptance' and source['source_partition'] == 'test_reserved'
        assert note['source_usable'] and note['request_scope'] == 'global'
        assert sha(source['source_image']) == source['source_sha256']
        image = DATA/'inputs'/f'native-accept-{i:03d}.png'
        save(image, load_rgb(source['source_image']))
        cases.append(dict(id=f'native-accept-{i:03d}', source_id=sid, group_id=source['group_id'], domain=note['domain'],
            split='acceptance-native', family=family, action='original-native-request', image=str(image.relative_to(ROOT)), input_sha256=sha(image),
            instructions={'en': source['instruction'], 'zh': zh}, expected_change=expected,
            must_preserve='主体、构图与未要求的色相和主要明暗细节；不凭空恢复原图已经丢失的内容，不能新增明显过曝、压黑、非预期肤色或无关改动。',
            failure_conditions=['方向相反、力度明显不足或过强', '复合要求漏项', '只有参数变化而无可见的要求效果', '新增重大过曝、压黑、偏色、失真或无关编辑'],
            instruction_origin='English original stored GIER annotation unchanged; Chinese AI translation; one semantic task per original group.',
            source_record_id=source['record_id'], source_partition='original test_reserved, never training_pool',
            original_source_sha256=source['source_sha256'], native_annotation=source['native_annotation'],
            source_classification_evidence=note['evidence'], no_reference_or_target_label=True))
    assert len(cases) == len({c['group_id'] for c in cases}) == 20
    assert Counter(c['domain'] for c in cases) == {'portrait': 10, 'landscape': 10}
    old_accept = {c['group_id'] for c in read(ROOT/'data/training/functional-v2-20261104/acceptance-final-cases.json')}
    assert not old_accept & {c['group_id'] for c in cases}
    write(dest, cases)
    write(DATA/'acceptance-native-freeze.json', dict(at=stamp(), status='source_requests_and_criteria_frozen_before_round6_training_or_output',
        cases=20, language_rows=40, domains={'portrait': 10, 'landscape': 10}, cases_sha256=sha(dest),
        source_isolation_sha256=sha(audit_path), source_conflict_resolution_sha256=sha(EXP/'planning/source-conflict-resolution.json') if excluded else None,
        source_manifest_sha256=sha(EXP/'source-extension/acceptance/manifest.json'),
        observations_sha256=sha(EXP/'source-extension/acceptance/observations.jsonl'), implementation_sha256=sha(__file__),
        overlap_existing_controlled_acceptance_groups=0,
        opening_rule='No model outputs until final weights, parser, prompt and editor are locked after developmental selection. No checkpoint replacement or parameter changes on these outcomes.',
        history='Source photographs directly viewed now to define tasks, not new unviewed photos. Original reserved test partition, no reference/output used in this freeze, no user feedback. Prior datasets have been evaluated historically; do not call it a pristine newly collected independent human test.',
        training_data_admitted=False, candidate_model_inference_started=False))
    print('Native acceptance20 photos/40 language rows frozen, model outputs remain closed')


if __name__ == '__main__': main()

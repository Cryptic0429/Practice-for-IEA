"""Freeze native developer requests/criteria from directly observed, isolated photos before inference."""
import json
from collections import Counter
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.editor import load_rgb, save
from .source_bank import EXP, DATA

# Picked for domain/task/state coverage from source images and original requests only.
# No candidate output from this round exists or is consulted.
SPECS = {
    1: ('brown_tone', '添加棕褐色色调。', '整体出现可辨认的棕褐色效果，保留人物及亮暗层次，不能仅添加黄绿色偏色。'),
    2: ('compound', '让颜色更鲜明，同时更暖一些。', '田野和建筑的色彩同时更鲜明、更暖，不能漏掉任一项或产生不自然荧光色。'),
    3: ('compound', '提高亮度和对比度。', '人物和画面可见提亮并增强对比，不能只做一项，衣物和面部仍有层次。'),
    4: ('saturation', '增强色彩，让颜色更鲜明、更有表现力。', '树木、河流和天空整体增艳，颜色仍自然，云层和山体细节保留。'),
    6: ('white_balance', '请减弱照片中的棕色调。', '减弱原有棕黄色偏色，人物与室内物体仍保持自然颜色，避免过度蓝青偏色。'),
    8: ('saturation', '增强色彩，让颜色更饱和。', '服装、头饰和背景可见增艳，保留自然肤色与衣服细节，避免过度饱和。'),
    11: ('compound', '稍微提高对比度和亮度，并增强色彩。', '轻微提高对比、提亮并增艳，三项均可见，保留自然肤色、天空与衣物纹理。'),
    13: ('lighting', '稍微提亮阴影。', '只需适度改善暗前景可见性，保留夕阳及亮天空，不能整体大幅过曝。'),
    16: ('lighting', '提亮这张照片。', '人物和室内画面整体可见提亮，面部及衣物仍自然，不增加大面积发白。'),
    23: ('contrast', '提高对比度。', '人像明暗分离可见增强，面部和头发仍有自然层次，避免额外偏色及压黑。'),
    29: ('compound', '把照片稍微调暗一点，并略微提高对比度。', '整体仅稍微调暗并增加对比，保留两人面部与衣服，不能大幅压黑或反向提亮。'),
    32: ('compound', '把照片调暗，并提高对比度。', '星空与山体整体可见收暗并增强明暗分离，同时保留可辨星点和山峰轮廓。'),
    40: ('lighting', '稍微调暗一点。', '幼童和室内场景轻微整体变暗，保留面部和浅色衣服层次，不应大幅变黑。'),
    67: ('lighting', '请提亮整张照片。', '人像及背景整体可见提亮，皮肤、头发和砖墙层次保持，不以仅改颜色代替。'),
    70: ('lighting', '大幅提高亮度。', '星空和山体可见明显提亮，保留星点及地平线，不把噪声放大到掩盖主要景物。'),
    100: ('saturation', '增强色彩，让颜色更鲜明、更有表现力。', '谷仓、草地和天空整体更鲜明，红色、绿色和蓝色仍自然，纹理保持。'),
    106: ('saturation', '让照片色彩更丰富、更饱和。', '草地与树木的颜色明显更鲜明，仍保留自然色相和明暗层次，不应泛黄或荧光绿。'),
    108: ('compound', '稍微提亮，增加对比度和色彩。', '画面适度提亮、增对比并增艳，码头及人物轮廓保留，不能让太阳附近大面积泛白。'),
    148: ('lighting', '请把照片提亮一点。', '梯田和山体温和整体提亮，天空、水面和山体层次保留，不能过度提亮。'),
    161: ('lighting', '把照片大幅调暗。', '天空、山峰与湖面明显调暗，保留山峰轮廓与倒影；人物可继续为剪影，不要求恢复面部。'),
}


def main():
    destination = DATA/'development-native-cases.json'
    if destination.exists():
        raise RuntimeError('Native developer cases already frozen')
    manifest = read(EXP/'source-review/manifest.json')
    bank = {r['id']: r for r in manifest['candidates']}
    notes = {r['id']: r for r in map(json.loads, (EXP/'source-review/observations.jsonl').read_text().splitlines())}
    selected = {f'development-native-{i:03d}' for i in SPECS}
    for file in ('source-candidate-audit.json', 'source-geometry-audit.json'):
        audit = read(EXP/'planning'/file)
        assert audit['manifest_sha256'] == sha(EXP/'source-review/manifest.json')
        for collision in audit['collisions']:
            if collision['reason'] in ('candidate_development_near_historical_training', 'development_near_historical_training'):
                assert collision['a'] not in selected, collision
            elif collision['reason'] in ('near_current_sources',):
                assert not (collision['a'] in selected or collision['b'] in selected), collision
    cases = []
    for i, (family, zh, expected) in SPECS.items():
        sid = f'development-native-{i:03d}'
        source, note = bank[sid], notes[sid]
        assert source['split'] == 'development' and note['source_usable'] and note['request_scope'] == 'global'
        assert note['domain'] in ('portrait', 'landscape')
        assert sha(source['source_image']) == source['source_sha256']
        image = DATA/'inputs'/f'native-dev-{i:03d}.png'
        save(image, load_rgb(source['source_image']))
        cases.append(dict(id=f'native-dev-{i:03d}', source_id=sid, group_id=source['group_id'],
            domain=note['domain'], split='development-native', family=family, action='original-native-request',
            image=str(image.relative_to(ROOT)), input_sha256=sha(image),
            instructions={'en': source['instruction'], 'zh': zh}, expected_change=expected,
            must_preserve='主体、构图、未要求的色相及主要明暗细节；已有缺失细节不要求凭空恢复，但不得新增明显过曝、压黑或非预期肤色偏移。',
            failure_conditions=['方向相反或只调用工具却没有可见效果', '力度明显不足或过强', '漏掉复合请求中的组成', '新增重大过曝、压黑、失真或无关改动'],
            instruction_origin='English stored original annotation unchanged, whitespace-normalized provenance match; Chinese AI translation; both form one semantic case.',
            source_record_id=source['record_id'], source_partition='original validation_reserved, never training_pool',
            original_source_sha256=source['source_sha256'], native_annotation=source['native_annotation'],
            source_classification_evidence=note['evidence'], no_reference_or_target_label=True))
    assert len(cases) == 20 and len({c['group_id'] for c in cases}) == 20
    assert Counter(c['domain'] for c in cases) == {'portrait': 10, 'landscape': 10}
    previous_dev = {c['group_id'] for c in read(ROOT/'data/training/functional-v2-20261104/development-final-cases.json')}
    old_native = {c['group_id'] for c in read(ROOT/'data/training/functional-v2-20261104/natural-use-final-cases.json')}
    assert not old_native & {c['group_id'] for c in cases}, 'Preserve old6 as separate regressions, not new probes'
    write(destination, cases)
    write(DATA/'development-native-freeze.json', dict(status='requests_criteria_and_sources_frozen_before_round6_model_output',
          at=stamp(), cases=20, language_rows=40, domains={'portrait': 10, 'landscape': 10},
          cases_sha256=sha(destination), source_manifest_sha256=sha(EXP/'source-review/manifest.json'),
          source_observations_sha256=sha(EXP/'source-review/observations.jsonl'),
          audit_sha256={f: sha(EXP/'planning'/f) for f in ('source-candidate-audit.json', 'source-geometry-audit.json')},
          overlapping_controlled_development_groups=sorted(previous_dev & {c['group_id'] for c in cases}),
          overlap_old6_native_groups=0, no_round6_model_inference_started=True,
          history='Original validation pool with historical evaluations/source viewing. Some source groups overlap controlled developer cases; do not sum as independent photos. It is not a pristine new human test.',
          implementation_sha256=sha(__file__), training_sources_not_yet_admitted=True,
          acceptance_status='Existing controlled acceptance closed; native acceptance source/criteria preparation still required.'))
    print('Native developer20 photos/20 tasks frozen; no model output or target label used')


if __name__ == '__main__':
    main()

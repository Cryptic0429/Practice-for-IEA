"""Freeze source-group-disjoint bilingual cases before any new model training."""
from __future__ import annotations
import collections
import hashlib
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from .common import ROOT, DATA, EXP, NAME, TOOLS, EDITOR_VERSION, read, write, sha, stamp
from .editor import load_rgb, edit, save, statistics

FAMILIES = ('lighting', 'white_balance', 'saturation', 'contrast', 'brown_tone')
TEXT = {
 'lighten': [
  ('提亮这张照片的暗部，保留亮处和天空的层次。', 'Lift the darker parts of this photo while preserving bright areas and sky detail.'),
  ('让画面明亮一些，阴影中的细节更清楚，避免过曝。', 'Make the photo a little brighter and reveal shadow detail without overexposure.'),
  ('适度提亮阴影，不要把高光变成一片白。', 'Moderately lift the shadows without turning the highlights white.'),
  ('暗的地方看不清，请把它们提亮，同时保留原有颜色。', 'The dark areas are hard to see. Brighten them while keeping the original colours.'),
  ('请改善这张照片的暗部光照，亮处不要变得刺眼。', 'Please improve the lighting in the dark areas without making the bright areas glaring.')],
 'darken': [
  ('适度降低画面亮度，让光线柔和一些，保留暗部细节。', 'Moderately darken the photo for softer light while retaining shadow detail.'),
  ('让照片暗一点，但不要把阴影压成纯黑。', 'Make the photograph a little darker without crushing the shadows.'),
  ('稍微收暗过亮的部分，保留画面原有的颜色和细节。', 'Tone down the bright areas slightly and preserve the original colours and details.'),
  ('请把曝光压低一些，让明亮的区域更柔和。', 'Please lower the exposure a little so the bright areas look softer.'),
  ('画面有点亮，请压暗一点，主体仍然需要清楚可见。', 'The photo looks a bit bright. Darken it a little while keeping the subject clearly visible.')],
 'warm': [
  ('把整体色调调暖一些，保持颜色自然。', 'Make the overall colour tone warmer while keeping it natural.'),
  ('给照片增加适度的暖色，避免过强的橙黄色。', 'Add moderate warmth to the photograph without a heavy orange-yellow cast.'),
  ('让这张照片温暖一点，不要改变亮度和内容。', 'Make this photo warmer without changing its brightness or content.'),
  ('请增加一点暖色调。', 'Please add a little warmth to the photo.'),
  ('想要温暖一些的色彩，画面其他部分保持自然。', 'I would like warmer colours while keeping the rest of the image natural.')],
 'cool': [
  ('把整体色调调冷一些，保持颜色自然。', 'Make the overall colour tone cooler while keeping it natural.'),
  ('给照片增加适度的冷色，避免过强的蓝色。', 'Add moderate coolness without a heavy blue cast.'),
  ('让这张照片冷一点，不要改变亮度和内容。', 'Make this photo cooler without changing its brightness or content.'),
  ('请增加一点冷色调。', 'Please add a little coolness to the photo.'),
  ('想要清冷一些的色彩，画面其他部分保持自然。', 'I would like cooler colours while keeping the rest of the image natural.')],
 'saturate': [
  ('让颜色鲜艳一些，避免过度饱和。', 'Make the colours more vivid without oversaturation.'),
  ('适度提高色彩饱和度，保留自然的色相和亮度。', 'Moderately increase colour saturation while retaining natural hues and brightness.'),
  ('颜色有点平淡，请增艳一些，不要过分夸张。', 'The colours are a little dull. Make them richer without exaggerating them.'),
  ('请让照片的色彩更丰富一点。', 'Please make the colours in this photo a little richer.'),
  ('增加一些色彩活力，同时保持照片真实自然。', 'Give the colours some extra life while keeping the photo realistic and natural.')],
 'desaturate': [
  ('让颜色柔和一些，适度降低饱和度，不要变成黑白。', 'Soften the colours by moderately reducing saturation, without making the photo black and white.'),
  ('降低一点色彩浓度，保留自然的颜色。', 'Reduce colour intensity a little while retaining natural colours.'),
  ('让照片的色彩淡雅一些，保持原来的明暗。', 'Make the colours more subdued while keeping the original lighting.'),
  ('请把颜色调淡一点，但仍然保留彩色。', 'Please tone down the colours a little, but keep the image in colour.'),
  ('想要低饱和度的自然色彩，亮度不需要改变。', 'I would like natural, less saturated colours without changing the brightness.')],
 'more_contrast': [
  ('适度增强对比度，保留亮部和暗部细节。', 'Moderately increase contrast while preserving highlight and shadow detail.'),
  ('让画面的明暗层次更分明，不要压黑阴影或过曝。', 'Make the tonal separation more distinct without crushing shadows or blowing highlights.'),
  ('增加一点对比度，保持颜色自然。', 'Add a little contrast while keeping the colours natural.'),
  ('照片有些平，请增强明暗对比，避免效果过强。', 'The photo looks a little flat. Increase tonal contrast without overdoing it.'),
  ('请让这张照片的对比更鲜明一些。', 'Please give this photograph a little more contrast.')],
 'less_contrast': [
  ('适度降低对比度，让明暗过渡更柔和。', 'Moderately reduce contrast for softer tonal transitions.'),
  ('让画面对比柔和一些，保留色彩和细节。', 'Soften the contrast while retaining colours and details.'),
  ('降低一点对比度，不要把画面弄得灰蒙蒙。', 'Lower the contrast a little without making the photo look washed out.'),
  ('这张照片想要柔和一点的明暗效果。', 'I would like softer tonal contrast in this photograph.'),
  ('请减弱明暗反差，同时保留清楚的内容。', 'Please reduce the tonal contrast while keeping the content clear.')],
 'brown': [
  ('给照片添加棕褐色复古色调，保留明暗层次和细节。', 'Give the photo a brown sepia vintage tone while preserving tonal detail.'),
  ('把整体颜色调成自然的棕褐色风格，避免高光过曝。', 'Apply a natural overall brown sepia style without blowing out highlights.'),
  ('想要棕色调的复古照片效果，不要只把画面提亮。', 'I want a vintage brown-toned photograph, not merely a brighter image.'),
  ('请添加棕色调。', 'Please, add brown tone.'),
  ('请把这张照片调成棕褐色风格，保留清楚的层次。', 'Please turn this photograph into a sepia-brown style with clear tonal detail.')],
 'identity': [
  ('保持这张图片原样，不要调整任何颜色或光照。', 'Keep this image exactly as it is. Do not adjust its colours or lighting.'),
  ('不要修改照片，直接保留原图。', 'Do not edit the photograph. Keep the original image.'),
  ('这张照片不需要修图，请保持不变。', 'This photograph needs no editing. Leave it unchanged.'),
  ('请不要做任何修改。', 'Please do not make any changes.'),
  ('保留原来的图片，不添加任何效果。', 'Retain the original image without adding any effects.')]
}

CRITERIA = {
 'lighting': ('光照按要求变亮或变暗；提亮要求应改善暗部可见性。', '亮部有层次，暗部不压黑，颜色和主体不被无关改变。'),
 'white_balance': ('色调按要求明显变暖或变冷。', '亮度和饱和度基本稳定，不出现过强橙黄、蓝或绿色偏。'),
 'saturation': ('颜色按要求更鲜艳或更柔和。', '保留原有色相和明暗，不出现荧光色或意外黑白。'),
 'contrast': ('明暗反差按要求增强或减弱。', '不明显丢失高光暗部细节，不出现灰雾或不必要偏色。'),
 'brown_tone': ('整体应呈现可辨认的棕褐色风格；绿色植被不能仍是主要色调。', '保留结构、亮部和暗部层次，不出现过曝或荧光绿。'),
 'identity': ('像素保持原样。', '没有任何不必要的编辑。')
}


def fingerprint(rgb):
    gray = cv2.cvtColor(cv2.resize(rgb, (64, 64)), cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    dct = cv2.dct(cv2.resize(gray, (32, 32)))[:8, :8].reshape(-1)[1:]
    bits = dct > np.median(dct)
    return sum(int(b) << i for i, b in enumerate(bits)), gray


def near(a, b):
    if (a[0] ^ b[0]).bit_count() > 6:
        return False
    x, y = a[1], b[1]
    return float(np.abs(x-y).mean()) < .01 or (min(x.std(), y.std()) > .005 and np.corrcoef(x.ravel(), y.ravel())[0,1] > .985)


def propose(case, rgb):
    action = case['action']
    stat = statistics(rgb)
    if action == 'identity':
        return [{}]
    if action == 'lighten':
        parameters = {'shadows': 55 if stat['mean_luminance'] > .3 else 65}
    elif action == 'darken':
        parameters = {'highlights': -40}
    elif action in ('warm', 'cool'):
        parameters = {'temperature': 25 if action == 'warm' else -25}
    elif action in ('saturate', 'desaturate'):
        parameters = {'saturation': 25 if action == 'saturate' else -25}
    elif action in ('more_contrast', 'less_contrast'):
        parameters = {'contrast': 20 if action == 'more_contrast' else -20}
    elif action == 'brown':
        parameters = {'sepia': 95}
    else:
        raise ValueError(action)
    candidates = [{k: min(100, max(-100, round(v*scale))) for k, v in parameters.items()}
                  for scale in (.75, 1, 1.25)]
    return list({json_key(p): p for p in candidates}.values())


def json_key(p):
    return __import__('json').dumps(p, sort_keys=True)


def render_candidates(cases):
    manifest = []
    for case in cases:
        rgb = load_rgb(ROOT / case['image'])
        if max(rgb.shape[:2]) > 1024:
            scale = 1024 / max(rgb.shape[:2])
            rgb = cv2.resize(rgb, (round(rgb.shape[1]*scale), round(rgb.shape[0]*scale)), interpolation=cv2.INTER_AREA)
        candidates = propose(case, rgb)
        entries = []
        for index, parameters in enumerate(candidates):
            image = edit(rgb, parameters)
            path = DATA / 'candidates-v1.2' / case['id'] / f'candidate-{index:02d}.png'
            save(path, image)
            entries.append({'candidate_id': path.stem, 'parameters': parameters,
                            'image': str(path.relative_to(ROOT)), 'sha256': sha(path),
                            'statistics': statistics(image),
                            'visual_preview_max_edge': 1024,
                            'origin': 'controlled_tool_proposal_not_original_reference_optimization'})
        manifest.append({'case_id': case['id'], 'candidates': entries})
    write(DATA / 'candidate-manifest.json', manifest)
    return manifest


def main():
    if (DATA / 'case-manifest.json').exists():
        raise RuntimeError('Frozen cases already exist; use the recorded manifest instead of regenerating them')
    DATA.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    sources = {
        'train': ROOT / 'data/training/quality-v2-20261103/train3k.json',
        'development': ROOT / 'data/training/quality-v2-20261103/validation.json',
        'acceptance': ROOT / 'data/stage1/splits/image-edit-3k-v1/test.json'}
    used = set()
    features = []
    all_cases = {}
    exclusions = []
    for split in ('acceptance', 'development', 'train'):
        by_group = {}
        for row in read(sources[split]):
            if split == 'train' and not row['extra_info']['instruction'].strip():
                continue
            by_group.setdefault(row['group_id'], row)
        pool = sorted(by_group.values(), key=lambda r: hashlib.sha256((NAME+split+r['group_id']).encode()).hexdigest())
        cases = []
        tasks = [('lighting', ('lighten', 'darken')), ('white_balance', ('warm', 'cool')),
                 ('saturation', ('saturate', 'desaturate')), ('contrast', ('more_contrast', 'less_contrast')),
                 ('brown_tone', ('brown',))]
        if split == 'train':
            tasks.append(('identity', ('identity',)))
        for family, actions in tasks:
            count = 25 if split == 'train' and len(actions) == 2 else (40 if split == 'train' and family == 'brown_tone' else (20 if split == 'train' else 10))
            chosen = 0
            for row in pool:
                gid = row['group_id']
                if gid in used:
                    continue
                rgb = load_rgb(row['images'][0])
                small = cv2.resize(rgb, (256, 256))
                stats = statistics(small)
                if family != 'identity' and (not .14 < stats['mean_luminance'] < .82 or stats['luminance_std'] < .055):
                    continue
                if family == 'saturation' and not .13 < stats['saturation'] < .75:
                    continue
                fp = fingerprint(small)
                if any(near(fp, f) for f in features):
                    exclusions.append({'split': split, 'group_id': gid, 'reason': 'near_duplicate_of_previously_selected_source'})
                    continue
                used.add(gid)
                features.append(fp)
                source_image = DATA / 'inputs' / f'{gid[:20]}.png'
                save(source_image, rgb)
                selected_actions = actions if split == 'train' and len(actions) == 2 and chosen < 15 else (actions[chosen % len(actions)],)
                pair_id = f'{split}-{family}-{chosen:03d}' if len(selected_actions) == 2 else None
                for action in selected_actions:
                    variant = chosen % 3 if split == 'train' else (3 if split == 'development' else 4)
                    zh, en = TEXT[action][variant]
                    case_id = f'{split}-{len(cases)+1:03d}'
                    expected, preserve = CRITERIA[family]
                    cases.append({'id': case_id, 'split': split, 'group_id': gid, 'family': family,
                                  'action': action, 'pair_id': pair_id,
                                  'image': str(source_image.relative_to(ROOT)), 'input_sha256': sha(source_image),
                                  'original_source_image': row['images'][0], 'original_source_sha256': sha(row['images'][0]),
                                  'original_record_id': row['id'],
                                  'original_request': row['extra_info']['instruction'],
                                  'instructions': {'zh': zh, 'en': en},
                                  'instruction_origin': 'controlled_bilingual_task' if not pair_id else 'controlled_bilingual_counterfactual_pair',
                                  'expected_change': expected, 'must_preserve': preserve,
                                  'failure_conditions': ['未完成要求或方向相反', '编辑幅度不足以完成要求',
                                                         '明显过曝、压黑、偏色、细节损失或无关变化'],
                                  'reference_usage': 'Original GIER target is NOT the target for this rewritten task.'})
                chosen += 1
                if chosen == count:
                    break
            if chosen != count:
                raise RuntimeError(f'Insufficient nonduplicate eligible groups for {split}/{family}: {chosen}/{count}')
        all_cases[split] = cases
        write(DATA / f'{split}-cases.json', cases)
    assert len(all_cases['train']) == 220
    assert len({r['group_id'] for r in all_cases['train']}) == 160
    assert len({r['pair_id'] for r in all_cases['train'] if r['pair_id']}) == 60
    assert all(len(all_cases[s]) == 50 for s in ('development', 'acceptance'))
    manifest = {'version': NAME, 'frozen_at': stamp(), 'status': 'cases_frozen_labels_not_yet_accepted',
                'editor': EDITOR_VERSION, 'tools': list(TOOLS),
                'splits': {s: {'cases': len(rows), 'language_records': 2*len(rows),
                               'source_groups': len({r['group_id'] for r in rows}),
                               'cases_sha256': sha(DATA / f'{s}-cases.json'),
                               'source': str(sources[s].relative_to(ROOT)), 'source_sha256': sha(sources[s]),
                               'families': dict(collections.Counter(r['family'] for r in rows))}
                           for s, rows in all_cases.items()},
                'cross_split_groups_disjoint': True, 'near_duplicate_exclusions': exclusions,
                'selection_rule': 'Seeded stable group order; explicit brightness/saturation eligibility; pHash and grayscale correlation duplicate screen.',
                'limits': ['All new task requests are controlled bilingual tasks, not untouched natural GIER requests.',
                           'Development and acceptance source photographs were evaluated historically; these are not never-viewed tests.',
                           'Targets for rewritten tasks are independently verified tool renders; original GIER targets are not reused.',
                           'Perceptual duplicate screening is heuristic, not proof of complete semantic independence.',
                           'All review is AI-assisted, without human validation.']}
    write(DATA / 'case-manifest.json', manifest)
    render_candidates(all_cases['train'])
    print(__import__('json').dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    import sys
    if '--render-only' in sys.argv:
        render_candidates(read(DATA/'train-cases.json'))
    else:
        main()

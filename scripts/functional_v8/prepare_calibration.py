"""Training-pool-only screening of an external editing reward adapter."""
import hashlib
import json
import math
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageOps
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.editor import load_rgb, save
from functional_v1.curate import font
from functional_v7.editor_soft import edit

EXP = ROOT / 'experiments/functional-v8-20261106'
DATA = ROOT / 'data/training/functional-v8-20261106'
REWARD_REVISION = 'dcce052accfc75f2d48279c889a71755ecdf737b'
REWARD_SHA = 'b78dc22fea2ebedfa17780a71ece05cec7062e6b738eeb3e2d21740cd44d7297'
CODE_REVISION = '4609c5d2ebb62fdebf665d3c924686d896ef1f74'


def main():
    assert not (EXP / 'calibration-preparation.json').exists()
    cases_file = ROOT / 'data/training/functional-v7-20261105/cases.json'
    admission_file = cases_file.with_name('admission.json')
    cases = read(cases_file)
    assert sha(cases_file) == read(admission_file)['cases_sha256']
    lookup = {c['id']: c for c in cases}
    old_exp = ROOT / 'experiments/functional-v7-20261105'
    full_file = old_exp / 'soft-label-candidates/full-resolution-observations.json'
    bad_notes = {n['case_id']: n for n in read(full_file)['observations']}
    quality_ids = [
        ('native-train-n013', 'native-train-n013-AI-brown-v7-r2'),
        ('native-train-n028', 'native-train-n028-AI-soft-contrast-v7'),
        ('native-train-n070', 'native-train-n070-AI-mild-light-v7'),
        ('native-train-n127', 'native-train-n127-AI-mild-light-v7'),
        ('native-train-n119', 'native-train-n119'),
        ('native-train-n186', 'native-train-n186'),
        ('native-train-x030', 'native-train-x030'),
        ('native-train-n190', 'native-train-n190-AI-muted-colour-v7')]
    quality, used_groups = [], set()
    for old_id, good_id in quality_ids:
        case, bad = lookup[good_id], bad_notes[old_id]
        assert case['action'] == 'edit' and bad['quality'] == 'minor_issue'
        assert case['input_sha256'] == bad['input_sha256'] == sha(ROOT / case['image'])
        assert sha(ROOT / case['target']) == case['target_sha256']
        assert sha(ROOT / bad['image']) == bad['image_sha256']
        used_groups.add(case['group_id'])
        quality.append(dict(id='quality-' + old_id, group_id=case['group_id'], domain=case['domain'],
            input_image=case['image'], input_sha256=case['input_sha256'],
            good_image=case['target'], good_sha256=case['target_sha256'],
            bad_image=bad['image'], bad_sha256=bad['image_sha256'],
            good_label_evidence=case['label_evidence'], bad_label_evidence=bad,
            scope='Image-quality-only pair; official PQ prompt has no instruction. Some images answer different requests. This never counts the failed original request as completed.',
            bad_severity='minor_issue_from_original_size_direct_AI_review_not_human'))
    assert sum(r['domain'] == 'portrait' for r in quality) == 4
    direction = []
    for family, tool, minimum in [('lighting', 'exposure', 25), ('white_balance', 'temperature', 15),
                                  ('saturation', 'saturation', 25), ('contrast', 'contrast', 25),
                                  ('brown_tone', 'sepia', 50)]:
        for domain in ('portrait', 'landscape'):
            eligible = [c for c in cases if c['action'] == 'edit' and c['family'] == family
                        and c['domain'] == domain and c['group_id'] not in used_groups
                        and set(c['parameters']) == {tool} and abs(c['parameters'][tool]) >= minimum]
            assert eligible, (family, domain)
            # Deterministic before any reward model predictions; no reward-based curation.
            case = min(eligible, key=lambda c: hashlib.sha256(('20261114:' + c['id']).encode()).hexdigest())
            used_groups.add(case['group_id'])
            before = load_rgb(ROOT / case['image'])
            assert sha(ROOT / case['image']) == case['input_sha256']
            wrong = {'temperature': -40, 'saturation': 25} if tool == 'sepia' else {tool: -60 if case['parameters'][tool] > 0 else 60}
            output = EXP / 'calibration-candidates' / (case['id'] + '-opposite.png')
            save(output, edit(before, wrong, 'bounded-fixed'))
            direction.append(dict(id='direction-' + case['id'], case_id=case['id'], group_id=case['group_id'],
                family=family, domain=domain, instructions=case['instructions'],
                input_image=case['image'], input_sha256=case['input_sha256'],
                good_image=case['target'], good_sha256=case['target_sha256'],
                bad_image=str(output.relative_to(ROOT)), bad_sha256=sha(output), bad_parameters=wrong,
                good_label_evidence=case['label_evidence'], new_bad_render_direct_review='pending'))
    sheets = []
    for row in direction:
        canvas = Image.new('RGB', (1800, 780), 'white'); draw = ImageDraw.Draw(canvas)
        draw.text((16, 12), row['id'], font=font(23), fill='black')
        for i, line in enumerate(textwrap.wrap(row['instructions']['en'], 120)):
            draw.text((16, 50 + i * 25), line, font=font(20), fill='black')
        # Labels identify review roles, never a model or reward prediction.
        for i, (label, path) in enumerate([('INPUT / identity candidate', row['input_image']),
                                           ('Previously admitted target', row['good_image']),
                                           ('New opposite-direction candidate', row['bad_image'])]):
            draw.text((i * 600 + 12, 130), label, font=font(20), fill='black')
            with Image.open(ROOT / path) as image:
                fitted = ImageOps.contain(image.convert('RGB'), (580, 600), Image.Resampling.LANCZOS)
            canvas.paste(fitted, (i * 600 + 10 + (580 - fitted.width) // 2, 170 + (600 - fitted.height) // 2))
        path = EXP / 'calibration-candidates' / (row['id'] + '.jpg')
        canvas.save(path, quality=95, subsampling=0)
        sheets.append(dict(id=row['id'], sheet=str(path.relative_to(ROOT)), sha256=sha(path)))
    write(DATA / 'calibration-candidates.json', dict(direction=direction, quality=quality, sheets=sheets))
    write(EXP / 'calibration-preparation.json', dict(at=stamp(), status='new_direction_images_pending_direct_review_no_GPU_started',
        source_cases_sha256=sha(cases_file), source_admission_sha256=sha(admission_file),
        cases_sha256=sha(DATA / 'calibration-candidates.json'),
        semantic_tasks=10, direction_comparisons=20, languages=['zh', 'en'], quality_pairs=8,
        original_photo_groups=len(used_groups), photo_domains={'portrait': 9, 'landscape': 9},
        data_scope='Only the already used round7 training160 photo groups. No developer or acceptance photos are loaded. Not an independent test; external model training overlap with GIER is unknown.',
        reward_model=dict(repository='EditScore/EditScore-7B', revision=REWARD_REVISION, adapter_sha256=REWARD_SHA,
                          source_repository='https://github.com/VectorSpaceLab/EditScore', source_revision=CODE_REVISION),
        initial_gates=dict(direction_semantic_tasks_required=9, direction_semantic_tasks_total=10,
                           each_task='Admitted result must beat both no-edit and opposite-direction candidates in BOTH languages; any tie/invalid counts as failure.',
                           quality_pairs_required=8, quality_pairs_total=8,
                           quality_rule='Admitted quality must strictly beat known damaged render under min(naturalness,artifacts); ties/invalid fail. These are minor defects, not a major-damage detection certification.'),
        future_use='Only if this fixed calibration passes may the reward model propose/rank training candidates. Direct Codex image review remains necessary. No automatic promotion or replacement of functional acceptance.',
        GPU_limit_seconds_each=1800, shared_gpu_hour_cap=16, own_policy_training_started=False,
        limitations='Small AI-labelled calibration, not human endorsement or proof of reward reliability. Official prompt assumes generated images although these are photographs; domain transfer explicitly tested.'))
    print(json.dumps(dict(direction_semantics=len(direction), quality_pairs=len(quality), groups=len(used_groups), GPU_started=False)))


if __name__ == '__main__':
    main()

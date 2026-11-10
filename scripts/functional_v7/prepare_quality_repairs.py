"""Additional training-label proposals triggered by recorded visual defects."""
import hashlib
from pathlib import Path

from functional_v1.common import ROOT, read, write, sha, stamp, validate
from functional_v1.editor import load_rgb, save
from .diagnose_editor import EXP
from .editor_soft import edit
from .prepare_label_repairs import panel


def main():
    out = EXP / 'label-quality-repairs'
    assert not (out / 'protocol.json').exists(), 'Keep earlier proposals immutable'
    source = ROOT / 'data/training/functional-v6-20261105/cases.json'
    cases = {c['id']: c for c in read(source)}
    proposals = {
        'train-214-style-contrast-strong': [
            dict(contrast=28, shadows=30, highlights=-15, saturation=-5),
            dict(contrast=35, shadows=35, highlights=-25, saturation=-8)],
        'native-train-n013': [
            dict(contrast=40, exposure=-30, saturation=15, shadows=25, highlights=-25),
            dict(contrast=50, exposure=-35, saturation=10, shadows=35, highlights=-35)],
        'native-train-n028': [
            dict(contrast=15, shadows=15, highlights=-10),
            dict(contrast=20, shadows=25, highlights=-15)],
        'native-train-n070': [
            dict(exposure=60, shadows=20),
            dict(exposure=45, shadows=30, saturation=-8)],
        'native-train-n127': [
            dict(exposure=65, shadows=35, highlights=-15, contrast=8),
            dict(exposure=80, shadows=20, highlights=-10)],
        'native-train-n119': [
            dict(exposure=30, shadows=35, highlights=-10),
            dict(exposure=45, shadows=20, highlights=-12)],
        'native-train-n186': [
            dict(exposure=20, shadows=55, highlights=-15),
            dict(exposure=30, shadows=40, highlights=-20)],
        'native-train-n190': [
            dict(contrast=35, saturation=25, highlights=-20, shadows=10),
            dict(contrast=45, saturation=25, highlights=-30, shadows=20)],
        'native-train-x030': [
            dict(exposure=55, shadows=30, highlights=-15),
            dict(exposure=70, shadows=15, highlights=-20)],
        'native-train-n029': [
            dict(exposure=-30, contrast=65, saturation=10, highlights=-25),
            dict(exposure=-35, contrast=85, saturation=5, highlights=-20)],
        'native-train-n093': [
            dict(exposure=-10, brightness=15, contrast=65, saturation=-15, temperature=35, tint=-15),
            dict(exposure=-15, brightness=20, contrast=80, saturation=-20, temperature=45, tint=-20)],
        'native-train-n068': [
            dict(exposure=-15, contrast=45, sepia=40, temperature=10, tint=-5, highlights=-35),
            dict(exposure=-20, contrast=55, sepia=50, temperature=10, tint=-10, highlights=-45)],
    }
    assert len(proposals) == 12
    for cid, values in proposals.items():
        assert cases[cid]['split'] == 'train'
        assert sha(ROOT / cases[cid]['image']) == cases[cid]['input_sha256']
        for parameters in values:
            validate(parameters)
    write(out / 'protocol.json', dict(
        at=stamp(), status='frozen_before_additional_quality_repair_outputs',
        source_cases_sha256=sha(source),
        source_full_observations_sha256=sha(EXP / 'soft-label-candidates/full-resolution-observations.json'),
        source_first_repairs_sha256=sha(EXP / 'label-repairs/direct-observations.json'),
        code_sha256=sha(__file__), editor_sha256=sha(Path(__file__).with_name('editor_soft.py')),
        proposals=proposals, inputs_and_original_bilingual_requests_unchanged=True,
        rationale='Visual followup found highlight/shadow compression and amplified source noise. Propose gentler shadow lifts or highlight protection. Earlier natural-request repairs for n029/n093 remain incomplete; try stronger tonal separation with less neon colour. This is not evidence that any candidate is usable.',
        selection='Direct visual review before admission; uncertainties remain excluded. Do not use L1. Do not declare reliable impossibility merely because a tested edit failed.',
        new_training_admitted=False, new_training_started=False, live_tools_changed=False,
        heldout_access=False, GPU_hours=0))
    old = {r['case_id']: r for r in read(EXP / 'soft-label-candidates/manifest.json')['cases']}
    records = []
    sheets = []
    for cid, values in proposals.items():
        c = cases[cid]
        rgb = load_rgb(ROOT / c['image'])
        items = []
        for p in values:
            result = edit(rgb, p, 'bounded-fixed')
            label = hashlib.sha256(result.tobytes()).hexdigest()[:12]
            path = out / 'images' / f'{cid}-{label}.png'
            save(path, result)
            records.append(dict(case_id=cid, label=label, parameters=p, input_image=c['image'],
                                input_sha256=c['input_sha256'], instructions=c['instructions'],
                                image=str(path.relative_to(ROOT)), image_sha256=sha(path),
                                kind='same_input_quality_repair', admitted=False))
            items.append((label, str(path.relative_to(ROOT))))
        assert len({label for label, _ in items}) == len(values)
        path = out / (cid + '.jpg')
        panel(path, cid, c['instructions']['en'],
              [('INPUT', c['image']), ('PREVIOUS CANDIDATE WITH QUALITY CONCERN', old[cid]['new_candidate'])] + sorted(items))
        sheets.append(dict(case_id=cid, sheet=str(path.relative_to(ROOT)), sha256=sha(path), kind='parameter_repair'))
    write(out / 'manifest.json', dict(at=stamp(), protocol_sha256=sha(out / 'protocol.json'),
                                      records=records, sheets=sheets, new_training_admission=False))
    print('Created 24 candidates for 12 original training requests. No admission or GPU work.')


if __name__ == '__main__':
    main()

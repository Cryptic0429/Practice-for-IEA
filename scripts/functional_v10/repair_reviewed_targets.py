"""Execute replacements only for explicitly reviewed quality concerns.

No automatic acceptance: every replacement remains pending direct viewing.
The original candidates and their negative observations are preserved.
"""
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.editor import load_rgb, save
from .prepare_counterfactuals import EXP, render, INSTRUCTIONS
from .extend_candidates import COMPOUND


def main():
    out = EXP / 'repairs/reviewed-portrait-batch-03-07-08-09'
    assert not (out / 'manifest.json').exists()
    initial = {c['id']: c for c in read(EXP / 'counterfactual-candidates/manifest.json')['cases']}
    extended = {c['id']: c for c in read(EXP / 'extended-candidates/manifest.json')['cases']}
    style = {c['id']: c for c in read(EXP / 'style-candidates/manifest.json')['cases']}
    planned = [
        (3, 'normal-compound', 'canonical', dict(exposure=10, contrast=10, shadows=15, saturation=5)),
        (7, 'normal-light', 'canonical', dict(exposure=40, highlights=-20)),
        (7, 'normal-compound', 'canonical', dict(exposure=10, contrast=10, shadows=15, saturation=8, highlights=-15)),
    ]
    for index in (8, 9):
        planned.extend([
            (index, 'normal-compound', 'canonical', dict(exposure=10, contrast=8, shadows=20, saturation=5)),
            (index, 'deep-compound', 'deep-clean-input', dict(exposure=20, contrast=5, shadows=35, saturation=5)),
            (index, 'mild-noise-compound', 'mild-noise-input', dict(exposure=20, contrast=5, shadows=35, saturation=5)),
        ])
    candidates = []
    for index, role, input_role, parameters in planned:
        cid = f'visual-risk-{index:02d}'
        original = style[cid] if role.startswith('normal-') else extended[cid]
        source_record = extended[cid]['images'][input_role] if input_role == 'mild-noise-input' else initial[cid]['images'][input_role]
        source = ROOT / source_record['path']
        assert sha(source) == source_record['sha256']
        old = original['images'][role]
        target = out / f'{cid}-{role}-v2.png'
        save(target, render(load_rgb(source), parameters))
        candidates.append(dict(id=f'{cid}-{role}-v2', group=cid, role=role,
            input=source_record, parameters=parameters, output=dict(path=str(target.relative_to(ROOT)), sha256=sha(target)),
            original=old, instructions=INSTRUCTIONS if role == 'normal-light' else COMPOUND,
            status='executed_pending_direct_review_not_admitted'))
    write(out / 'manifest.json', dict(at=stamp(), candidates=candidates,
        reasons={'03': 'Over-emphasized orange skin and vivid red paint in normal compound.',
                 '07': 'Normal lift/compound flattened extra bright white baby-chest/dress texture.',
                 '08': 'Compound candidates intensified already warm skin and hair towards yellow/orange.',
                 '09': 'Compound candidates intensified face and hand towards orange.'},
        old_pixels_and_observations_preserved=True, automatic_selection=False, L1_selection=False))
    print(f'Executed {len(candidates)} replacements; none admitted.')


if __name__ == '__main__':
    main()

"""CPU-only replacement attempts after changed labels failed direct review."""
import os
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.editor import load_rgb, save
from .outcomes import execute
from .prepare_labels import EXP, panel


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    folder = EXP / 'label-candidates/repairs/lighten-contrast'
    assert not (folder / 'manifest.json').exists()
    cases = {c['id']: c for c in read(EXP / 'candidate-cases.json')}
    rows, sheets = [], []
    for cid, candidates in {
        'native-train-n048': [dict(exposure=8, contrast=20, shadows=65),
                              dict(contrast=20, shadows=80)],
        'native-train-x250': [dict(exposure=10, contrast=20, shadows=65, highlights=-8),
                              dict(exposure=5, contrast=28, shadows=90, highlights=-8)],
    }.items():
        c = cases[cid]
        rgb = load_rgb(ROOT / c['image'])
        images = [('INPUT', ROOT / c['image'])]
        ids = []
        for i, parameters in enumerate(candidates, 1):
            rid = f'{cid}-repair{i}'
            output = folder / (rid + '.png')
            save(output, execute(rgb, dict(action='edit', parameters=parameters))['result'])
            rows.append(dict(id=rid, case_id=cid, parameters=parameters,
                             input=dict(path=c['image'], sha256=c['input_sha256']),
                             output=dict(path=str(output.relative_to(ROOT)), sha256=sha(output)),
                             status='executed_not_viewed_not_admitted'))
            images.append((rid, output)); ids.append(rid)
        sheet = panel(cid, c['instructions']['en'], images, folder / f'{cid}.jpg')
        sheets.append(dict(case_id=cid, candidates=ids, panel=sheet))
    write(folder / 'manifest.json', dict(at=stamp(), candidates=rows, panels=sheets,
          reason='Original changed labels increased contrast but did not visibly lighten. Preserve raw failed labels and review replacements separately.',
          GPU_hours=0, training_started=False))


if __name__ == '__main__':
    main()

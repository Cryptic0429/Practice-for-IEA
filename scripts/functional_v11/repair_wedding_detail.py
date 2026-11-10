"""CPU attempt protecting dress folds after two brighter labels flattened them."""
import os
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.editor import load_rgb, save
from .prepare_labels import EXP, panel
from .outcomes import execute


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    folder = EXP / 'label-candidates/repairs/wedding-detail'
    assert not (folder / 'manifest.json').exists()
    c = next(c for c in read(EXP / 'candidate-cases.json') if c['id'] == 'native-train-n048')
    parameters = dict(exposure=5, contrast=15, shadows=60, highlights=-8)
    output = folder / 'native-train-n048-repair3.png'
    save(output, execute(load_rgb(ROOT/c['image']), dict(action='edit', parameters=parameters))['result'])
    row = dict(id='native-train-n048-repair3', case_id=c['id'], parameters=parameters,
               input=dict(path=c['image'], sha256=c['input_sha256']),
               output=dict(path=str(output.relative_to(ROOT)), sha256=sha(output)))
    sheet = panel(c['id'], c['instructions']['en'], [('INPUT', ROOT/c['image']), ('REPAIR3', output)], folder/'panel.jpg')
    write(folder/'manifest.json', dict(at=stamp(), candidates=[row], panel=sheet,
          GPU_hours=0, status='executed_not_viewed_not_admitted'))


if __name__ == '__main__':
    main()

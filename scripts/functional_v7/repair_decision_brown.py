"""One stronger brown-tone proposal for an already reviewed new AI request."""
import copy
from pathlib import Path
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.editor import load_rgb, save
from .diagnose_editor import EXP
from .editor_soft import edit
from .prepare_decision_candidates import pair


def main():
    out = EXP / 'decision-brown-repair'
    assert not (out / 'protocol.json').exists()
    source = EXP / 'decision-candidates/manifest.json'
    c = copy.deepcopy(next(c for c in read(source)['records'] if c['id'] == 'native-train-n013-AI-brown-v7'))
    assert sha(ROOT / c['image']) == c['input_sha256']
    assert c['instruction_origin'].startswith('AI-authored')
    c.update(id=c['id'] + '-r2', parameters=dict(sepia=85), decision=dict(action='edit', parameters=dict(sepia=85)),
             target=None, target_sha256=None, admitted=False)
    write(out / 'protocol.json', dict(at=stamp(), status='frozen_before_brown_label_repair_output',
                                      source_manifest_sha256=sha(source),
                                      source_observations_sha256=sha(EXP / 'decision-candidates/direct-observations.json'),
                                      code_sha256=sha(__file__), editor_sha256=sha(Path(__file__).with_name('editor_soft.py')),
                                      candidate=c, GPU_hours=0, admitted=False,
                                      disclosure='Same new AI brown request and original input; not the excluded original GIER contrast request.'))
    path = out / 'image.png'
    save(path, edit(load_rgb(ROOT / c['image']), c['parameters'], 'bounded-fixed'))
    c.update(target=str(path.relative_to(ROOT)), target_sha256=sha(path))
    panel = out / 'panel.jpg'
    pair(panel, c, c['target'], 'STRONGER BROWN CANDIDATE | not admitted')
    write(out / 'manifest.json', dict(candidate=c, sheet=str(panel.relative_to(ROOT)), sheet_sha256=sha(panel), admitted=False))
    print('One stronger brown candidate rendered; no admission.')


if __name__ == '__main__':
    main()

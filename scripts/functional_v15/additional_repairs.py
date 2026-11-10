"""CPU-only proposals for the additional failure found in native inspection."""
import os
from PIL import Image, ImageDraw, ImageOps
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.curate import font
from .editor import load_rgb, save
from .outcomes import execute, validate_decision, EDITOR_VERSION

EXP = ROOT / 'experiments/functional-v15-20261106'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    cid = 'native-train-x250'
    source = EXP / 'label-candidates/manifest.json'
    item = next(i for i in read(source)['items'] if i['id'] == cid)
    import json
    notes = EXP / 'label-native-observations.jsonl'
    note = next(n for n in map(json.loads, notes.read_text().splitlines()) if n['id'] == cid)
    assert note['quality'] == 'm' and sha(ROOT / item['input']['path']) == item['input']['sha256']
    folder = EXP / 'label-repairs/wave2' / cid
    assert not folder.exists()
    folder.mkdir(parents=True)
    rgb = load_rgb(ROOT / item['input']['path'])
    recipes = [dict(exposure=35, contrast=14, shadows=10, highlights=-10),
               dict(exposure=45, contrast=12, highlights=-15)]
    results = []
    for index, parameters in enumerate(recipes, 1):
        label = f'R{index}'
        decision = dict(action='edit', parameters=parameters)
        validate_decision(decision)
        outcome = execute(rgb, decision)
        assert outcome['status'] == 'edited'
        path = folder / (label + '.png')
        save(path, outcome['result'])
        results.append(dict(candidate=label, decision=decision, image=str(path.relative_to(ROOT)),
                            image_sha256=sha(path), label_admitted=False))
    canvas = Image.new('RGB', (1800, 820), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), cid + ' | CPU REPAIR PROPOSALS, NOT MODEL OUTPUT', font=font(22), fill='black')
    draw.text((12, 55), item['instructions']['en'], font=font(20), fill='black')
    for n, (label, path) in enumerate([('INPUT', ROOT / item['input']['path'])] +
                                     [(r['candidate'], ROOT / r['image']) for r in results]):
        draw.text((n * 600 + 12, 150), label, font=font(21), fill='black')
        with Image.open(path) as im:
            fitted = ImageOps.contain(im.convert('RGB'), (580, 600), Image.Resampling.LANCZOS)
        canvas.paste(fitted, (n * 600 + 10 + (580 - fitted.width) // 2,
                              190 + (600 - fitted.height) // 2))
    sheet = folder / 'review.jpg'
    canvas.save(sheet, quality=96, subsampling=0)
    row = dict(id=cid, input=item['input'], instructions=item['instructions'], results=results,
               prior_candidate=item['candidate'], failed_observation=note,
               sheet=str(sheet.relative_to(ROOT)), sheet_sha256=sha(sheet), label_admitted=False)
    write(folder.parent / 'manifest.json', dict(at=stamp(), items=[row],
          source_manifest_sha256=sha(source), source_native_observations_sha256=sha(notes),
          editor=EDITOR_VERSION, editor_sha256=sha(ROOT / 'scripts/functional_v15/editor.py'),
          executed_edits=2, GPU_hours=0,
          policy='Unchanged literal request; explicit AI proposals for observed shadow chroma failure. '
                 'No automatic admission; direct overview and native quality inspection required.'))
    print('Executed 2 additional proposals; none admitted.')


if __name__ == '__main__':
    main()

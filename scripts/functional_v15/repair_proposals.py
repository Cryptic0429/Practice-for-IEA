"""Render an explicit finite repair recipe file; never rank or admit results."""
import argparse, json, os, textwrap
from PIL import Image, ImageDraw, ImageOps
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.curate import font
from .editor import load_rgb, save
from .outcomes import execute, validate_decision, EDITOR_VERSION

EXP = ROOT / 'experiments/functional-v15-20261106'


def main(bundle, recipes_path):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    recipes_path = ROOT / recipes_path
    proposal = read(recipes_path)
    problem = ROOT / proposal['failure_observations']
    assert sha(problem) == proposal['failure_observations_sha256']
    observed = list(map(json.loads, problem.read_text().splitlines()))
    source = EXP / 'label-candidates/manifest.json'
    bank = {i['id']: i for i in read(source)['items']}
    dest = EXP / 'label-repairs' / bundle
    assert not dest.exists()
    dest.mkdir(parents=True)
    items = []
    for cid, recipes in proposal['recipes'].items():
        assert any(n['id'] == cid and (n['quality'] != 'A' or n['completion'] != 'C') for n in observed)
        item = bank[cid]
        assert sha(ROOT / item['input']['path']) == item['input']['sha256']
        rgb = load_rgb(ROOT / item['input']['path'])
        out = dest / cid; out.mkdir()
        results = []
        for index, parameters in enumerate(recipes, 1):
            label = f'R{index}'; decision = dict(action='edit', parameters=parameters)
            validate_decision(decision)
            result = execute(rgb, decision); assert result['status'] == 'edited'
            path = out / (label + '.png'); save(path, result['result'])
            results.append(dict(candidate=label, decision=decision, image=str(path.relative_to(ROOT)),
                                image_sha256=sha(path), label_admitted=False))
        canvas = Image.new('RGB', (600 * (1 + len(results)), 820), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), cid + ' | UNCHANGED REQUEST | CPU PROPOSALS, NOT MODEL OUTPUT', font=font(22), fill='black')
        for j, line in enumerate(textwrap.wrap(item['instructions']['en'], 130)):
            draw.text((12, 45 + 25 * j), line, font=font(20), fill='black')
        for n, (label, path) in enumerate([('INPUT', ROOT / item['input']['path'])] +
                                         [(r['candidate'], ROOT / r['image']) for r in results]):
            draw.text((n * 600 + 12, 150), label, font=font(21), fill='black')
            with Image.open(path) as im:
                fitted = ImageOps.contain(im.convert('RGB'), (580, 600), Image.Resampling.LANCZOS)
            canvas.paste(fitted, (n * 600 + 10 + (580 - fitted.width) // 2,
                                  190 + (600 - fitted.height) // 2))
        sheet = out / 'review.jpg'; canvas.save(sheet, quality=96, subsampling=0)
        items.append(dict(id=cid, input=item['input'], instructions=item['instructions'], results=results,
            prior_candidate=item['candidate'], sheet=str(sheet.relative_to(ROOT)), sheet_sha256=sha(sheet),
            label_admitted=False))
    write(dest / 'manifest.json', dict(at=stamp(), items=items, source_manifest_sha256=sha(source),
        explicit_recipe_file=str(recipes_path.relative_to(ROOT)), explicit_recipe_sha256=sha(recipes_path),
        failure_observations=proposal['failure_observations'], failure_observations_sha256=sha(problem),
        editor=EDITOR_VERSION, editor_sha256=sha(ROOT / 'scripts/functional_v15/editor.py'),
        executed_edits=sum(len(i['results']) for i in items), GPU_hours=0,
        policy='Explicit finite AI proposals for observed failures; unchanged requests. No automatic admission or L1 ranking.'))
    print('Executed repair bundle', bundle, 'cases', len(items))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--bundle', required=True)
    parser.add_argument('--recipes', required=True); args = parser.parse_args()
    main(args.bundle, args.recipes)

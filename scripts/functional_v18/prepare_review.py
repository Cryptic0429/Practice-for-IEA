"""Render exact crossed labels and blind them with actual model results."""
import hashlib
import os
import textwrap

from PIL import Image, ImageDraw, ImageOps
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.curate import font
from functional_v15.editor import load_rgb
from functional_v15.outcomes import execute

EXP = ROOT / 'experiments/functional-v18-20261107'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert read(EXP / 'status.json')['status'] == 'completed'
    protocol = read(EXP / 'diagnostic-protocol.json')
    import json
    predictions = {row['case_id']: row for row in [json.loads(s) for s in (EXP / 'predictions.jsonl').read_text().splitlines()]}
    out = EXP / 'review'
    out.mkdir(exist_ok=False)
    items = []
    for pair in protocol['pairs']:
        canvas = Image.new('RGB', (2000, 1160), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), 'TRAIN-ONLY ANONYMOUS COUNTERFACTUAL COMPARISON', fill='black', font=font(23))
        for n, line in enumerate(textwrap.wrap(pair['cases'][0]['instructions']['en'], 150)):
            draw.text((12, 45+n*25), line, fill='black', font=font(20))
        case_items = []
        for side, case in enumerate(pair['cases']):
            rgb = load_rgb(ROOT / case['image'])
            assert sha(ROOT / case['image']) == case['input_sha256']
            pred = predictions[case['id']]
            other = pair['cases'][1-side]
            sources = []
            for origin, decision in [('own_label', case['decision']), ('crossed_label', other['decision'])]:
                execution = execute(rgb, decision)
                assert execution['result'] is not None
                path = out / (case['id']+'-'+origin+'.png')
                Image.fromarray(execution['result']).save(path, compress_level=1)
                sources.append(dict(origin=origin, decision=decision, path=str(path.relative_to(ROOT)), sha256=sha(path), status=execution['status']))
            sources.append(dict(origin='actual_model', decision=pred['decision'], path=pred['output_image'], sha256=pred['output_sha256'], status=pred['status']))
            sources.sort(key=lambda row: hashlib.sha256(('r18-blind:'+case['id']+':'+row['origin']).encode()).hexdigest())
            mapping = {label: source for label, source in zip('ABC', sources)}
            opaque = hashlib.sha256(case['id'].encode()).hexdigest()[:10]
            y = 130 + side*505
            draw.text((12, y), 'Row '+opaque, fill='black', font=font(21))
            assets = []
            for col, (label, source) in enumerate([('INPUT', dict(path=case['image'], sha256=case['input_sha256']))]+list(mapping.items())):
                x = col*500
                draw.text((x+12, y+30), label, fill='black', font=font(22))
                if source['path'] is None:
                    draw.text((x+18, y+220), 'NO RESULT: '+source['status'], fill='black', font=font(22))
                    continue
                path = ROOT / source['path']
                assert sha(path) == source['sha256']
                image = Image.fromarray(load_rgb(path))
                w, h = image.size
                assert max(w,h)<=1600, 'Explicit critical-region protocol required for larger source'
                fitted = ImageOps.contain(image, (480, 415), Image.Resampling.LANCZOS)
                canvas.paste(fitted, (x+10+(480-fitted.width)//2, y+65+(415-fitted.height)//2))
                native = out / (opaque+'-'+label+'-native.png')
                image.save(native, compress_level=1)
                with Image.open(native) as check:
                    assert check.size==image.size and check.tobytes()==image.tobytes()
                assets.append(dict(label=label, path=str(native.relative_to(ROOT)), sha256=sha(native),
                    source=source['path'], source_sha256=source['sha256'], size=[w,h], region='whole_unscaled_RGB'))
            case_items.append(dict(case_id=case['id'], opaque_row=opaque, instructions=case['instructions'],
                input_sha256=case['input_sha256'], mapping=mapping, native=assets))
        sheet = out / (pair['id']+'.jpg')
        canvas.save(sheet, quality=96, subsampling=0)
        items.append(dict(pair_id=pair['id'], family=pair['family'], domain=pair['domain'],
            sheet=str(sheet.relative_to(ROOT)), sheet_sha256=sha(sheet), cases=case_items))
    write(out / 'manifest.json', dict(at=stamp(), protocol_sha256=sha(EXP/'diagnostic-protocol.json'), items=items,
        scope='12 TRAIN source pairs only. Own/crossed/model anonymous. Parameters not shown on sheets. No automatic preferences, new labels or user certification. Native files are exact whole unscaled RGB; thumbnails alone cannot verify detail quality.'))
    print('Prepared12 anonymous pair sheets,24 input cases, exact native assets.')


if __name__ == '__main__':
    main()

"""Render already adjudicated predictions for the report; no model or grading."""
from PIL import Image, ImageDraw, ImageFont
from functional_v1.common import ROOT, read, write, sha
from .visual_review import EXP, EVAL, case_file
import json


def main():
    dest = EXP / 'report-panels'
    dest.mkdir(exist_ok=True)
    manifest = []
    for split, ids in [('native-development', ['001','003','006','008','011','023','032','040','100','108','161']),
                       ('old-native', ['030'])]:
        cases = read(case_file(split))
        predictions = {model: {(r['case_id'],r['language']):r for r in map(json.loads,
            (EVAL / split / model / 'predictions.jsonl').read_text().splitlines())}
            for model in ['parent-new-manual','language-merger-e1']}
        for number in ids:
            case = next(c for c in cases if c['id'].endswith('-'+number))
            sources = [('INPUT', ROOT/case['image'])]
            rows = []
            for model in predictions:
                row = predictions[model][case['id'],'zh']; rows.append(row)
                sources.append((model, ROOT/row['output_image'] if row['status']=='executed' else None))
            canvas = Image.new('RGB', (1320, 690), 'white'); draw=ImageDraw.Draw(canvas)
            draw.text((12,12),case['id']+' | '+case['instructions']['en'],fill='black',font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18))
            for i,(label,path) in enumerate(sources):
                x=440*i
                draw.text((x+12,55),label,fill='black')
                if path:
                    im=Image.open(path).convert('RGB'); im.thumbnail((416,580))
                    canvas.paste(im,(x+12+(416-im.width)//2,92+(580-im.height)//2))
                else:
                    row=rows[i-1]
                    draw.text((x+12,310),row['status'],fill='black')
                    draw.text((x+12,340),str(row.get('decision') or row.get('error'))[:58],fill='black')
            target=dest/(case['id']+'.jpg');canvas.save(target,quality=94)
            manifest.append(dict(case_id=case['id'],split=split,path=str(target.relative_to(ROOT)),sha256=sha(target),
                language='zh',input_sha256=case['input_sha256'],predictions=rows,
                presentation='Whole frames with identical letterboxes; thumbnails only, no colour adjustment; not new visual ratings.'))
    write(dest/'manifest.json',manifest)
    print('Rendered',len(manifest),'report panels')


if __name__=='__main__':main()

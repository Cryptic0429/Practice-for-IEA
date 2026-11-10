"""Present actual predictions with their existing AI grades; never assign new grades."""
import argparse
import json
import os
import textwrap
from PIL import Image,ImageDraw,ImageOps
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.curate import font
from .visual_review import EXP,EVAL,case_file


def main(supplement=False):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    out=EXP/('report-panels-supplement' if supplement else 'report-panels');assert not (out/'manifest.json').exists()
    out.mkdir(parents=True,exist_ok=True)
    selected={'native-development':['native-dev-'+s for s in ('001','006','011','023','032','070','108','161')],
              'development':['development-006-restore-exposure-strong','development-027-compound',
                  'development-028-restore-saturation-mild','development-040-restore-saturation-mild',
                  'development-001-restore-exposure-strong']}
    if supplement:selected={'native-development':['native-dev-100'],'development':['development-028-restore-saturation-mild']}
    items=[]
    for split,ids in selected.items():
        cases={c['id']:c for c in read(case_file(split))}
        models=('parent-frozen','language-merger-e1','all-language-merger-e1')
        predictions={m:{(p['case_id'],p['language']):p for p in map(json.loads,(EVAL/split/m/'predictions.jsonl').read_text().splitlines())} for m in models}
        reviews={m:{(p['case_id'],p['language']):p['assessment'] for p in read(EVAL/split/m/'direct-review.json')} for m in models}
        for cid in ids:
            language='en' if supplement and split=='development' else 'zh'
            case=cases[cid];assert sha(ROOT/case['image'])==case['input_sha256']
            canvas=Image.new('RGB',(2000,690),'white');draw=ImageDraw.Draw(canvas)
            draw.text((16,12),cid+' | ACTUAL MODEL OUTPUTS | AI review, not user approval',font=font(22),fill='black')
            for n,line in enumerate(textwrap.wrap(case['instructions']['en'],145)):
                draw.text((16,47+24*n),line,font=font(19),fill='black')
            photos=[('INPUT',ROOT/case['image'])];notes={}
            for m,title in zip(models,('PARENT R11','R13 NARROW e1','R13 WIDE e1')):
                row=predictions[m][cid,language];grade=reviews[m][cid,language]
                label=title+' | '+grade['completion']+'/'+grade['quality']
                image=ROOT/row['output_image'] if row['status']=='executed' else None
                if image:assert sha(image)==row['output_sha256']
                photos.append((label,image));notes[m]=dict(status=row['status'],decision=row.get('decision'),assessment=grade,output=row.get('output_image'),output_sha256=row.get('output_sha256'))
            for n,(label,path) in enumerate(photos):
                for j,line in enumerate(textwrap.wrap(label,47)):
                    draw.text((n*500+12,120+22*j),line,font=font(16),fill='black')
                if path:
                    with Image.open(path) as im:image=ImageOps.contain(im.convert('RGB'),(480,470),Image.Resampling.LANCZOS)
                    canvas.paste(image,(n*500+10+(480-image.width)//2,180+(470-image.height)//2))
                else:draw.text((n*500+60,350),'NO RESULT IMAGE: DECLINED',font=font(18),fill='black')
            draw.text((16,661),'Equal-scale full frames, no colour adjustment. Presentation thumbnails; native-pixel QA remains in the review records.',font=font(17),fill='black')
            path=out/(cid+'.jpg');canvas.save(path,quality=95,subsampling=0)
            items.append(dict(case_id=cid,language=language,split=split,family=case['family'],instruction=case['instructions'],input=case['image'],models=notes,panel=str(path.relative_to(ROOT)),sha256=sha(path)))
    write(out/'manifest.json',dict(at=stamp(),items=items,rule='Actual frozen predictions and existing direct AI assessments; Chinese outputs displayed, bilingual completion reported separately. No new assessment or retouching.',GPU_hours=0))
    print('Created',len(items),'actual comparison panels')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--supplement',action='store_true');main(parser.parse_args().supplement)

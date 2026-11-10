"""Illustrate every native task family from already-final reviews, without regrading."""
import json,textwrap
from PIL import Image,ImageDraw,ImageOps
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.curate import font
EXP=ROOT/'experiments/functional-v21-20261107'

def main():
    selection=read(EXP/'development-selection.json');model=selection['models']['functional-v21-quality-20261205']
    out=EXP/'report-panels';out.mkdir(exist_ok=False)
    source=ROOT/read(EXP/'training-protocol.json')['sources']['native_development']['path']
    cases=read(source);folder=EXP/'evaluation'/'functional-editor-v15-shared-shadow-gain-candidate'/'native-development'/model['evaluation_id']
    notes={(r['case_id'],r['language']):r['assessment'] for r in read(folder/'direct-review.json')}
    rows={(r['case_id'],r['language']):r for r in map(json.loads,(folder/'predictions.jsonl').read_text().splitlines())}
    def seriousness(c):
        ranks=[]
        for language in ('zh','en'):
            n=notes[c['id'],language]
            ranks.append(4 if n['quality']=='major_issue' else 3 if n['completion']=='failed' else
                         2 if n['completion']=='partial' or n['quality']=='minor_issue' else
                         1 if n['completion']=='uncertain' or n['quality']=='uncertain' else 0)
        return max(ranks)
    records=[]
    for family in sorted({c['family'] for c in cases}):
        c=sorted((c for c in cases if c['family']==family),key=lambda c:(-seriousness(c),c['id']))[0]
        canvas=Image.new('RGB',(1800,770),'white');d=ImageDraw.Draw(canvas)
        d.text((16,8),family+' | '+c['id']+' | locked R21 seed1 e20 (replication failed)',fill='black',font=font(24))
        for i,line in enumerate(textwrap.wrap(c['instructions']['en'],140)):
            d.text((16,42+24*i),line,fill='black',font=font(20))
        images=[('INPUT',ROOT/c['image'],None)]
        for language in ('zh','en'):
            r=rows[c['id'],language];n=notes[c['id'],language]
            label=language.upper()+' | '+n['completion']+' / '+n['quality']
            images.append((label,ROOT/r['output_image'] if r['output_image'] else None,language))
        for i,(label,path,language) in enumerate(images):
            x=i*600;d.text((x+10,125),label,fill='black',font=font(19))
            if path:
                with Image.open(path) as im:fitted=ImageOps.contain(im.convert('RGB'),(580,550),Image.Resampling.LANCZOS)
                canvas.paste(fitted,(x+10+(580-fitted.width)//2,175+(550-fitted.height)//2))
            else:d.text((x+22,340),'No edited image: model declined',fill='black',font=font(21))
        d.text((16,739),'Full frames scaled equally; report illustration, not a new visual grade. Sources and final notes in manifest.',fill='black',font=font(17))
        file=out/(family+'.jpg');canvas.save(file,quality=95,subsampling=0)
        records.append(dict(family=family,case_id=c['id'],instructions=c['instructions'],panel=str(file.relative_to(ROOT)),panel_sha256=sha(file),
            input=dict(path=c['image'],sha256=c['input_sha256']),results={language:dict(path=rows[c['id'],language]['output_image'],
                sha256=rows[c['id'],language]['output_sha256'],status=rows[c['id'],language]['status'],assessment=notes[c['id'],language]) for language in ('zh','en')}))
    write(EXP/'report-panels.json',dict(at=stamp(),selection_rule='One case per native family, prioritizing nonacceptable final outcomes; lexicographic case ID breaks ties. Show both languages, never cherry-pick one language. Report only, not checkpoint selection.',
        model_id=model['model_id'],controller_sha256=model['controller_sha256'],cases_sha256=sha(source),records=records,new_assessments=0,GPU_calls=0))
    print('Created',len(records),'actual-model family panels from existing final evidence; no new grades or GPU.')

if __name__=='__main__':main()

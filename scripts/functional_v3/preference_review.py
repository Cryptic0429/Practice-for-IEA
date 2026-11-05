"""Inspect actual training-only output alternatives; never infer preference from numeric distance."""
import argparse,hashlib,json,math,textwrap
from .common import ROOT,ROUND2_DATA,EXP,DATA,read,write,sha,stamp


def prepare():
    from PIL import Image,ImageDraw,ImageOps
    from functional_v1.curate import font
    assert read(EXP/'collection/status.json')['status']=='completed'
    rows=list(map(json.loads,(EXP/'collection/predictions.jsonl').read_text().splitlines()))
    cases=read(ROUND2_DATA/'train-final-cases.json');assert len(rows)==720
    out=EXP/'preference-review';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Do not overwrite review bundle')
    sheets=[];same=[];invalid=[]
    for c in cases:
        assert c['split']=='train' and sha(ROOT/c['target'])==c['target_sha256']
        predictions=[r for r in rows if r['case_id']==c['id']]
        assert len(predictions)==2
        unique={}
        for r in predictions:
            if r['status']!='executed':invalid.append(r);continue
            assert sha(ROOT/r['output_image'])==r['output_sha256']
            if r['output_sha256']==c['target_sha256']:same.append({'case_id':c['id'],'language':r['language']});continue
            unique.setdefault(r['output_sha256'],{'image':r['output_image'],'sha256':r['output_sha256'],'parameters':r['parameters'],'origin':'policy','languages':[]})['languages'].append(r['language'])
        if not unique:continue
        unique[c['target_sha256']]={'image':c['target'],'sha256':c['target_sha256'],'parameters':c['parameters'],'origin':'approved_target'}
        entries=sorted(unique.values(),key=lambda r:hashlib.sha256((c['id']+r['sha256']).encode()).hexdigest())
        cols=2 if len(entries)==2 else 3;size=(cols*600,130+610*math.ceil((len(entries)+1)/cols));canvas=Image.new('RGB',size,'white');d=ImageDraw.Draw(canvas)
        d.text((12,8),c['id'],fill='black',font=font(23))
        for j,line in enumerate(textwrap.wrap(c['instructions']['en'],45*cols)):d.text((12,45+28*j),line,fill='black',font=font(22))
        for i,(label,path) in enumerate([('INPUT',c['image'])]+[(f'Output {i+1}',r['image']) for i,r in enumerate(entries)]):
            x,y=(i%cols)*600,130+(i//cols)*610;d.text((x+12,y),label,fill='black',font=font(23));im=ImageOps.contain(Image.open(ROOT/path).convert('RGB'),(580,550));canvas.paste(im,(x+10+(580-im.width)//2,y+40+(550-im.height)//2))
        file=out/(c['id']+'.jpg');canvas.save(file,quality=96,subsampling=0)
        sheets.append({'case_id':c['id'],'domain':c['domain'],'source_id':c['source_id'],'sheet':str(file.relative_to(ROOT)),'sha256':sha(file),'instructions':c['instructions'],'expected_change':c['expected_change'],'must_preserve':c['must_preserve'],'entries':entries})
    write(out/'manifest.json',{'at':stamp(),'sheets':sheets,'pixel_identical_target_rows':same,'invalid_rows':invalid,'policy':'Direct visual comparison. Only clear better completion/quality becomes a preference. Output difference is not itself an error or preference. Same language-agnostic render/case reviewed once.'})
    print(json.dumps({'sheets':len(sheets),'identical_rows':len(same),'invalid':len(invalid)}))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare']);a=ap.parse_args();prepare()
if __name__=='__main__':main()

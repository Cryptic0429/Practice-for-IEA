"""Truthful report panels, generated only after direct grading."""
import json,textwrap
from PIL import Image,ImageDraw,ImageOps
from .common import ROOT,EXP,EVAL,HELDOUT_DATA,read,write,sha
from functional_v1.curate import font

def main():
 out=EXP/'report-panels';out.mkdir(exist_ok=True);manifest=[]
 for split,cid in [('development','development-055-restore-exposure-strong'),('development','development-040-style-sepia-strong'),('natural-use','natural-019'),('natural-use','natural-022')]:
  case=next(c for c in read(HELDOUT_DATA/f'{split}-final-cases.json') if c['id']==cid)
  entries=[('INPUT',case['image'])];receipt=[]
  for mid in ['plain-e1','measured-e1']:
   rows=list(map(json.loads,(EVAL/split/mid/'predictions.jsonl').read_text().splitlines()))
   for lang in ['zh','en']:
    r=next(r for r in rows if r['case_id']==cid and r['language']==lang);assert r['status']=='executed';assert sha(ROOT/r['output_image'])==r['output_sha256'];entries.append((mid+' / '+lang.upper(),r['output_image']));receipt.append(r)
  canvas=Image.new('RGB',(1800,1350),'white');d=ImageDraw.Draw(canvas);d.text((15,10),cid,font=font(27),fill='black')
  for j,line in enumerate(textwrap.wrap(case['instructions']['en'],110)):d.text((15,49+j*27),line,font=font(23),fill='black')
  for i,(label,path) in enumerate(entries):
   x,y=i%3*600,130+i//3*600;d.text((x+12,y),label,font=font(23),fill='black');im=ImageOps.contain(Image.open(ROOT/path).convert('RGB'),(580,550));canvas.paste(im,(x+10+(580-im.width)//2,y+35+(550-im.height)//2))
  file=out/(cid+'.jpg');canvas.save(file,quality=96,subsampling=0);manifest.append({'sheet':str(file.relative_to(ROOT)),'sha256':sha(file),'case_id':cid,'actual_outputs':receipt,'presentation':'Full frames at identical scale; no exposure/color/crop change; model names revealed after direct assessment.'})
 write(out/'manifest.json',manifest)
if __name__=='__main__':main()

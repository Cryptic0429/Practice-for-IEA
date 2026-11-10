"""Explicit CPU proposals for actually observed failures; no automatic admission."""
import os,textwrap
from PIL import Image,ImageDraw,ImageOps
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.curate import font
from .editor import load_rgb,save
from .outcomes import execute,EDITOR_VERSION,validate_decision

EXP=ROOT/'experiments/functional-v15-20261106'
PROPOSALS={
 'native-train-n011':[dict(exposure=40,shadows=15,saturation=-8,temperature=-4,tint=4),dict(exposure=55,shadows=10,saturation=-12,temperature=-3,tint=6)],
 'native-train-n204':[dict(exposure=100,saturation=-15,highlights=-10),dict(exposure=70,brightness=20,shadows=10,highlights=-20,saturation=-18)],
 'native-train-x030':[dict(exposure=90,brightness=20,shadows=25,highlights=-10),dict(exposure=100,brightness=10,shadows=40,highlights=-12)],
 'native-train-x186':[dict(exposure=35,shadows=25,highlights=-10,tint=-8,saturation=-5),dict(exposure=55,shadows=10,highlights=-12,tint=-10,saturation=-10)],
 'r13-train-extension-081':[dict(exposure=30,shadows=10,saturation=-10),dict(exposure=40,shadows=5,temperature=-5,tint=-30,saturation=-10)],
 'r13-train-extension-538':[dict(temperature=18,tint=-20,saturation=3,contrast=14,shadows=5),dict(temperature=12,tint=-30,contrast=16,shadows=5)],
 'r13-train-native-073':[dict(exposure=50,brightness=35,contrast=16,shadows=10,highlights=-15,saturation=12),dict(exposure=70,brightness=20,contrast=20,shadows=15,highlights=-20,saturation=12)],
 'r13-train-native-074':[dict(exposure=80,brightness=25,shadows=20,highlights=-12,saturation=8),dict(exposure=100,brightness=15,shadows=10,highlights=-15,saturation=10)],
 'r13-train-native-214':[dict(exposure=55,brightness=5,shadows=10,contrast=14,saturation=8,highlights=-15),dict(exposure=65,contrast=12,saturation=8,highlights=-20)],
 'r13-train-native-234':[dict(exposure=35,shadows=10,saturation=-8,highlights=-12),dict(exposure=50,shadows=5,saturation=-12,highlights=-15)],
}


def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 original=EXP/'label-candidates/manifest.json';bank={i['id']:i for i in read(original)['items']}
 notes=EXP/'label-native-observations.jsonl'
 import json
 observed={n['id']:n for n in map(json.loads,notes.read_text().splitlines())}
 dest=EXP/'label-repairs/wave1';assert not (dest/'manifest.json').exists();items=[]
 for cid,recipes in PROPOSALS.items():
  item=bank[cid];note=observed[cid]
  assert note['completion']!='C' or note['quality']!='A'
  assert sha(ROOT/item['input']['path'])==item['input']['sha256']
  rgb=load_rgb(ROOT/item['input']['path']);out=dest/cid;out.mkdir(parents=True,exist_ok=True);results=[]
  for index,p in enumerate(recipes,1):
   label=f'R{index}';decision=dict(action='edit',parameters=p);validate_decision(decision)
   result=execute(rgb,decision);assert result['status']=='edited'
   path=out/(label+'.png');assert not path.exists();save(path,result['result'])
   results.append(dict(candidate=label,decision=decision,image=str(path.relative_to(ROOT)),image_sha256=sha(path),label_admitted=False))
  canvas=Image.new('RGB',(1800,820),'white');draw=ImageDraw.Draw(canvas)
  draw.text((12,10),cid+' | UNCHANGED REQUEST | CPU REPAIR PROPOSALS, NOT MODEL OUTPUT',font=font(22),fill='black')
  for j,line in enumerate(textwrap.wrap(item['instructions']['en'],135)):
   draw.text((12,45+25*j),line,font=font(20),fill='black')
  for n,(label,path) in enumerate([('INPUT',ROOT/item['input']['path'])]+[(r['candidate'],ROOT/r['image']) for r in results]):
   draw.text((n*600+12,150),label,font=font(21),fill='black')
   with Image.open(path) as im: fitted=ImageOps.contain(im.convert('RGB'),(580,600),Image.Resampling.LANCZOS)
   canvas.paste(fitted,(n*600+10+(580-fitted.width)//2,190+(600-fitted.height)//2))
  sheet=out/'review.jpg';assert not sheet.exists();canvas.save(sheet,quality=96,subsampling=0)
  items.append(dict(id=cid,input=item['input'],instructions=item['instructions'],results=results,
   prior_candidate=item['candidate'],failed_observation=note,sheet=str(sheet.relative_to(ROOT)),sheet_sha256=sha(sheet),label_admitted=False))
 write(dest/'manifest.json',dict(at=stamp(),items=items,source_manifest_sha256=sha(original),
  source_native_observations_sha256=sha(notes),editor=EDITOR_VERSION,editor_sha256=sha(ROOT/'scripts/functional_v15/editor.py'),
  executed_edits=sum(len(i['results']) for i in items),GPU_hours=0,
  policy='Only actual observed failures; unchanged literal requests. Parameters are explicit AI proposals. Every adopted result requires direct overview and native quality review; no L1 ranking or automatic admission. All failed originals remain intact.'))
 print('Executed',len(items),'repair cases /',sum(len(i['results']) for i in items),'proposals; none admitted.')


if __name__=='__main__':main()

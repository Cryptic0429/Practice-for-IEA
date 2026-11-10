"""CPU-only explicit parameter proposals for already observed failures.

Requests and failed pixels stay intact; no weakening of native instructions and
no result admission until direct visual review of the actual execution.
"""
import textwrap
from PIL import Image,ImageDraw,ImageOps
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb,save
from functional_v1.curate import font
from functional_v11.outcomes import execute,validate_decision,EDITOR_VERSION
from .source_candidates import EXP


PROPOSALS={
 'train-native-088':[dict(exposure=70,contrast=12),dict(exposure=85,contrast=16,temperature=3,tint=5)],
 'train-native-078':[dict(exposure=80,contrast=12,saturation=8),dict(exposure=95,contrast=16,saturation=8)],
 'train-native-184':[dict(exposure=5,contrast=20),dict(exposure=12,contrast=28)],
 'train-native-228':[dict(exposure=55),dict(exposure=70,contrast=4)],
 'train-native-151':[dict(exposure=65),dict(exposure=80,contrast=5)],
 'train-native-040':[dict(exposure=30),dict(exposure=45,contrast=3)],
 'train-native-219':[dict(exposure=65),dict(exposure=80,contrast=5)],
 'train-native-016':[dict(temperature=-16,tint=30,saturation=-15),dict(temperature=-22,tint=45,saturation=-18)],
 'train-native-069':[dict(temperature=10,tint=35,saturation=-8,contrast=8),dict(temperature=12,tint=45,saturation=-12,contrast=12)],
 'train-native-092':[dict(exposure=60,temperature=-12,tint=12,saturation=-18),dict(exposure=75,temperature=-10,tint=15,saturation=-20)],
 'train-native-205':[dict(exposure=60,contrast=18,saturation=8),dict(exposure=75,contrast=24,saturation=8)],
 'train-extension-396':[dict(exposure=12,contrast=8),dict(exposure=20,contrast=12)],
 'train-extension-121':[dict(exposure=55,contrast=25),dict(exposure=70,contrast=32)],
 'train-extension-339':[dict(exposure=20),dict(exposure=12,shadows=12)],
}


def main():
 dest=EXP/'primary-repairs';assert not (dest/'manifest.json').exists()
 original=EXP/'primary-candidates/manifest.json';bank={i['id']:i for i in read(original)['items']}
 items=[]
 for sid,recipes in PROPOSALS.items():
  item=bank[sid];assert sha(ROOT/item['input_image'])==item['input_sha256']
  rgb=load_rgb(ROOT/item['input_image']);out=dest/sid;out.mkdir(parents=True,exist_ok=True)
  results=[]
  for index,parameters in enumerate(recipes,1):
   label=f'R{index}';decision=dict(action='edit',parameters=parameters);validate_decision(decision)
   result=execute(rgb,decision);assert result['status']=='edited'
   path=out/(label+'.png');assert not path.exists();save(path,result['result'])
   results.append(dict(candidate=label,decision=decision,status=result['status'],image=str(path.relative_to(ROOT)),image_sha256=sha(path)))
  canvas=Image.new('RGB',(1320,820),'white');draw=ImageDraw.Draw(canvas)
  draw.text((12,10),sid+' | same request, explicit repair proposals',font=font(22),fill='black')
  for j,line in enumerate(textwrap.wrap(item['instructions']['en'],110)[:3]):draw.text((12,42+24*j),line,font=font(19),fill='black')
  for index,(label,path) in enumerate([('INPUT',ROOT/item['input_image'])]+[(r['candidate'],ROOT/r['image']) for r in results]):
   x=index*440;draw.text((x+10,115),label,font=font(21),fill='black')
   im=ImageOps.contain(Image.open(path).convert('RGB'),(420,650));canvas.paste(im,(x+10+(420-im.width)//2,155+(650-im.height)//2))
  sheet=out/'review.jpg';canvas.save(sheet,quality=96,subsampling=0)
  items.append(dict(item,results=results,sheet=str(sheet.relative_to(ROOT)),sheet_sha256=sha(sheet),repair_of=str(original.relative_to(ROOT)),label_admitted=False))
 write(dest/'manifest.json',dict(at=stamp(),items=items,editor=EDITOR_VERSION,GPU_hours=0,executed_edits=sum(len(i['results']) for i in items),
  source_manifest_sha256=sha(original),policy='Actual CPU execution, same input and unchanged requests. Manual repair proposals, not model predictions or admitted labels. Full-frame sheets with no colour changes; native-pixel followup required for noise/skin/highlight risks. Failed first candidates remain preserved.'))
 print('Executed explicit repairs',len(items))


if __name__=='__main__':main()

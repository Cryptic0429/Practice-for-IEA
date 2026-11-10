"""Compare only complete fixed-input sets; render illustrative witnesses on CPU."""
import json,collections
from PIL import Image,ImageDraw,ImageOps
from .common import *
from functional_v1.common import EXP as BUDGET
from functional_v1.curate import font
from functional_v15.outcomes import execute,EDITOR_VERSION
from functional_v1.editor import load_rgb,save
DIR=EXP/'export-robustness'
def rows(f):return list(map(json.loads,f.read_text().splitlines()))
def action(d):return (d['action'],d.get('reason'))
def signed(d):return tuple(sorted((k,1 if v>0 else -1 if v<0 else 0) for k,v in d.get('parameters',{}).items()))
def evidence_source(mid):
 if mid=='parent-R15':return ROOT/'experiments/functional-v15-20261106/evaluation'/EDITOR_VERSION/'native-development/language-merger-e1/predictions.jsonl'
 ex=EXP if mid.startswith('functional-v23') else ROOT/'experiments/functional-v22-20261107';meta=read(ex/'development-selection.json')['models'][mid]
 return ex/'evaluation'/EDITOR_VERSION/'native-development'/meta['evaluation_id']/'predictions.jsonl'
def pair(name,label,left,right):
 with Image.open(left) as i:a=i.convert('RGB')
 with Image.open(right) as i:b=i.convert('RGB')
 assert a.size==b.size;w,h=a.size;canvas=Image.new('RGB',(2*w,h+70),'white');d=ImageDraw.Draw(canvas)
 d.text((12,8),label,fill='black',font=font(20));d.text((12,38),'INPUT (native pixels of this file)',fill='black',font=font(18));d.text((w+12,38),'EXECUTED OUTPUT (same pixels / scale)',fill='black',font=font(18));canvas.paste(a,(0,70));canvas.paste(b,(w,70));canvas.save(name)
 assert canvas.crop((0,70,w,h+70)).tobytes()==a.tobytes() and canvas.crop((w,70,2*w,h+70)).tobytes()==b.tobytes()
def main():
 p=read(DIR/'protocol.json');run=read(DIR/'execution-status.json');r=rows(DIR/'generations.jsonl');assert len(r)==run['generations_saved']==293
 ids={*p['models'],*p['heads']};by=collections.defaultdict(dict);orig={};source_hashes={}
 for mid in sorted(ids):
  path=evidence_source(mid);orig[mid]={(x['case_id'],x['language']):x for x in rows(path)};source_hashes[str(path.relative_to(ROOT))]=sha(path)
 for x in r:
  assert x['model_id'] in ids and x['model_id'] not in by[x['id']] and x['status']=='parsed' and sha(ROOT/x['image'])==x['image_sha256'];by[x['id']][x['model_id']]=x
 complete={k:v for k,v in by.items() if set(v)==ids};unpaired={k:sorted(v) for k,v in by.items() if k not in complete};flips=[];counts={};examples=[]
 for mid in sorted(ids):
  c=collections.Counter();groups=collections.defaultdict(list)
  for v in complete.values():
   x=v[mid];before=orig[mid][x['case_id'],x['language']];assert before['instruction']==x['instruction'] and before['input_sha256']==x['original_sha256'];od=before['decision'];nd=x['decision'];groups[x['case_id'],x['condition']].append(x['language']);c['paired_language_inputs']+=1;c[nd['action']]+=1
   if action(od)!=action(nd):c['action_or_reason_changes']+=1;flips.append(dict(model_id=mid,case_id=x['case_id'],condition=x['condition'],language=x['language'],original=od,exported=nd,input=x['image']))
   elif signed(od)!=signed(nd):c['signed_toolset_changes']+=1
   elif od!=nd:c['magnitude_only_changes']+=1
   else:c['exact_decision_same']+=1
  c['bilingual_photo_condition_pairs']=sum(sorted(ls)==['en','zh'] for ls in groups.values());counts[mid]=dict(c)
 out=DIR/'witnesses';out.mkdir(exist_ok=False)
 for v in complete.values():
  first=v['parent-R15']
  if first['case_id']!='native-dev-070' or first['language']!='zh':continue
  decisions={mid:x['decision'] for mid,x in v.items()};editors=[mid for mid,d in decisions.items() if d['action']=='edit'];declines=[mid for mid,d in decisions.items() if d['action']=='decline'];assert len(editors)==3 and len(declines)==2
  outputs=[];source=ROOT/first['image'];img=load_rgb(source)
  for mid in editors:
   result=execute(img,decisions[mid]);f=out/f'{first["condition"]}-{mid}.png';save(f,result['result']);outputs.append(f)
  assert len({sha(f) for f in outputs})==1
  panel=out/f'{first["condition"]}-pair.png';pair(panel,first['condition']+' | 3 equal exposure edits; both R22 heads declined',source,outputs[0])
  examples.append(dict(case_id=first['case_id'],condition=first['condition'],language='zh',input=first['image'],decisions=decisions,executed_images={str(f.relative_to(ROOT)):sha(f) for f in outputs},declines_have_no_result=declines,panel=str(panel.relative_to(ROOT)),panel_sha256=sha(panel)))
 mid='functional-v23-quality-20261208';night=orig[mid]['native-dev-070','zh'];assert night['output_sha256']==orig['functional-v23-quality-20261207']['native-dev-070','zh']['output_sha256']
 panel=EXP/'report-panels/night-original-native-pixels.png';pair(panel,'Night input | both R23 seeds: exposure +65; quality disputed',ROOT/night['input_image'],ROOT/night['output_image'])
 trial=read(EXP/'demo/http-integration.json');t=[x for x in trial['checks'] if x['case_id']=='historical-case12-brown'];assert len(t)==2
 paths=[BUDGET/'demo/requests'/t[0]['request_id']/'input.png']+[BUDGET/'demo/requests'/x['request_id']/'output.png' for x in t]
 canvas=Image.new('RGB',(1800,695),'white');d=ImageDraw.Draw(canvas);d.text((12,10),'Historical case-12 regression: Please, add brown tone',fill='black',font=font(25))
 for i,(caption,path) in enumerate(zip(['INPUT','R23 seed1: sepia 85','R23 seed2: sepia 95'],paths)):
  d.text((600*i+10,60),caption,fill='black',font=font(22))
  with Image.open(path) as image:
   im=ImageOps.contain(image.convert('RGB'),(580,550));canvas.paste(im,(600*i+10+(580-im.width)//2,95+(550-im.height)//2))
 case12=EXP/'report-panels/case12-http.jpg';canvas.save(case12,quality=95,subsampling=0)
 write(DIR/'summary.json',dict(at=stamp(),planned_inputs=p['planned_inputs'],actual_predictions=len(r),full_LM_generations=run['actual_LM_generations_saved'],full_model_sets=len(complete),incomplete_sets=unpaired,
  full_sets_by_condition=dict(collections.Counter(next(iter(v.values()))['condition'] for v in complete.values())),models=counts,action_or_reason_changes=flips,original_prediction_file_hashes=source_hashes,
  visual_witnesses=examples,original_night_panel=str(panel.relative_to(ROOT)),original_night_panel_sha256=sha(panel),case12_panel=str(case12.relative_to(ROOT)),case12_panel_sha256=sha(case12),historical_case12_http_request_ids=[x['request_id'] for x in t],
  policy='Comparison is action/tool/sign/parameter stability on existing ten DEV photos with altered exports. It is not a functional quality score or independent evaluation. Only complete five-model rows compared; partial final rows retained. CPU witnesses executed after GPU stop. No changed thresholds, checkpoints or labels.'))
 print('Complete fixed 5-model sets',len(complete),'partial',unpaired,'action changes',len(flips),'native night witnesses',len(examples))
if __name__=='__main__':main()

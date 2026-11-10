"""Execute explicitly AI-authored paired requests on new real training inputs.

The original failed requests/targets are preserved and excluded. These new
requests have separate IDs and need fresh visual admission, never inherited
from a reference or from the passing primary task on the same photograph.
"""
import json,textwrap
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb,save
from functional_v1.curate import font
from functional_v11.outcomes import execute,validate_decision,EDITOR_VERSION
from .source_candidates import EXP

E={'train-native-'+x for x in ('234','139','033','183','230','060','125','181','123','085')}
E|={'train-extension-'+x for x in ('072','183','420','414','294','386','299','533','551','505')}
C={'train-native-'+x for x in ('117','212','122','175','236','170','184','130','214','118','223','144')}
C|={'train-extension-'+x for x in ('263','269','065','262','422','081','526','446')}
S={'train-native-'+x for x in ('056','078','092','044','073','192','132','074','108','216','114','094','051','134','006','065')}
S|={'train-extension-'+x for x in ('211','377','395','521')}


def dump_image(path,rgb):
 if path.exists():assert np.array_equal(load_rgb(path),rgb)
 else:save(path,rgb)


def main():
 dest=EXP/'paired-candidates';assert not (dest/'manifest.json').exists()
 primary={r['id']:r for r in read(EXP/'primary-candidates/manifest.json')['items']}
 repairs={r['id']:r for r in read(EXP/'primary-repairs/manifest.json')['items']}
 selected=read(EXP/'provisional-source-selection.json')['items']
 assert len(selected)==78 and Counter(x['domain'] for x in selected)==dict(portrait=39,landscape=39)
 selected += [dict(id='train-native-088',domain='portrait',bundle='new-AI-primary'),dict(id='train-native-065',domain='landscape',bundle='new-AI-primary')]
 selected.sort(key=lambda i:(0 if i['domain']=='portrait' else 1,list(primary).index(i['id'])))
 assert len(E)==len(C)==len(S)==20 and not(E&C or E&S or C&S)
 assert E|C|S<={i['id'] for i in selected}
 items=[];sheets=[];selection=[];family_counts=Counter();domain_index=Counter()
 new_primary={
  'train-native-088':dict(instructions=dict(en='Give the whole beach portrait a gentle warmer tone while preserving the existing brightness, people and water detail.',zh='让整张海边人像略偏暖，保留现有明暗、人物和水面细节。'),parameters=dict(temperature=8,tint=10)),
  'train-native-065':dict(instructions=dict(en='Reduce the overall colour saturation noticeably but keep the sunset brightness, shoreline and visible detail.',zh='明显降低整体色彩饱和度，保留日落的明暗、海岸线和现有细节。'),parameters=dict(saturation=-18)),
 }
 for index,choice in enumerate(selected):
  sid=choice['id'];source=primary[sid];out=dest/sid;out.mkdir(parents=True,exist_ok=True)
  assert sha(ROOT/source['input_image'])==source['input_sha256'];rgb=load_rgb(ROOT/source['input_image'])
  family='exposure' if sid in E else 'contrast' if sid in C else 'saturation' if sid in S else 'temperature'
  family_counts[family]+=1;local=[]
  if sid in new_primary:
   n=new_primary[sid];local.append(dict(id=sid+'-new-primary',instructions=n['instructions'],decision=dict(action='edit',parameters=n['parameters']),kind='separately_AI_authored_primary_after_original_failure',family='warm_style' if sid.endswith('088') else 'desaturation'))
  for direction in (-1,1):
   v=(8,12,16)[index%3] if family!='temperature' else (4,6,8)[index%3]
   if family=='exposure':
    en=('Brighten' if direction>0 else 'Darken')+' the whole photograph a little. Keep its colours, framing and visible detail.'
    zh=('稍微提亮' if direction>0 else '稍微压暗')+'整张照片，保留色彩、构图和现有细节。'
   elif family=='contrast':
    en=('Increase' if direction>0 else 'Soften')+' the overall contrast a little. Keep the framing and visible fine structure.'
    zh=('稍微增强' if direction>0 else '稍微减弱')+'整体对比度，保留构图和现有细小结构。'
   elif family=='saturation':
    en=('Increase' if direction>0 else 'Reduce')+' the overall colour saturation slightly, keeping the brightness and scene unchanged.'
    zh=('稍微增加' if direction>0 else '稍微降低')+'整体色彩饱和度，保留明暗和场景。'
   else:
    en='Give the whole photograph a slightly '+('warmer' if direction>0 else 'cooler')+' colour tone. Keep the brightness, framing and existing details.'
    zh='让整张照片的色调略偏'+('暖' if direction>0 else '冷')+'，保留明暗、构图和现有细节。'
   parameters={family:direction*v}
   if family=='temperature':parameters['tint']=-direction*v
   local.append(dict(id=sid+'-paired-'+('positive' if direction>0 else 'negative'),instructions=dict(en=en,zh=zh),
    decision=dict(action='edit',parameters=parameters),kind='AI_new_explicit_reverse_request',family=family,paired_direction=direction))
  pictures=[('INPUT',ROOT/source['input_image'],'Original training-pool image')]
  if choice['bundle']!='new-AI-primary':
   old=(primary if choice['bundle']=='primary-candidates' else repairs)[sid]
   r=next(x for x in old['results'] if x['candidate']==choice['chosen'])
   selection.append(dict(choice,primary_case_id=sid,primary_manifest=choice['bundle']+'/manifest.json',primary_decision=r['decision'],primary_image=r['image'],primary_image_sha256=r['image_sha256']))
  for c in local:
   decision=validate_decision(c['decision']);result=execute(rgb,decision);assert result['status']=='edited'
   path=out/(c['id']+'.png');dump_image(path,result['result'])
   c.update(source_id=sid,domain=source['domain'],group_id=source['group_id'],source_image=source['source_image'],source_sha256=source['source_sha256'],
    input_image=source['input_image'],input_sha256=source['input_sha256'],instruction_origin='AI_new_global_request',original_instruction=source['original_instruction'],native_annotation=source['native_annotation'],
    results=[dict(candidate='executed',decision=decision,status='edited',image=str(path.relative_to(ROOT)),image_sha256=sha(path))],label_admitted=False)
   pictures.append((c['id'].removeprefix(sid+'-'),path,c['instructions']['en']))
   if c['kind']=='separately_AI_authored_primary_after_original_failure':selection.append(dict(choice,primary_case_id=c['id'],primary_manifest='paired-candidates/manifest.json',primary_decision=decision,primary_image=str(path.relative_to(ROOT)),primary_image_sha256=sha(path)))
  width=440*len(pictures);canvas=Image.new('RGB',(width,950),'white');draw=ImageDraw.Draw(canvas)
  draw.text((12,10),sid+' | '+source['domain']+' | NEW AI requests, executed labels, not model predictions',font=font(21),fill='black')
  for j,(label,path,request) in enumerate(pictures):
   x=j*440;draw.text((x+10,48),label,font=font(20),fill='black')
   for line_index,line in enumerate(textwrap.wrap(request,43)[:5]):draw.text((x+10,79+23*line_index),line,font=font(17),fill='black')
   im=ImageOps.contain(Image.open(path).convert('RGB'),(420,650));canvas.paste(im,(x+10+(420-im.width)//2,270+(650-im.height)//2))
  sheet=out/'review.jpg';canvas.save(sheet,quality=96,subsampling=0)
  for c in local:c.update(sheet=str(sheet.relative_to(ROOT)),sheet_sha256=sha(sheet));items.append(c)
  sheets.append(dict(source_id=sid,domain=source['domain'],family=family,sheet=str(sheet.relative_to(ROOT)),case_ids=[c['id'] for c in local]))
  if domain_index[source['domain']]<10:
   decision=dict(action='keep');result=execute(rgb,decision);assert result['status']=='unchanged' and np.array_equal(result['result'],rgb)
   path=out/'keep.png';dump_image(path,result['result'])
   c=dict(source,id=sid+'-keep',source_id=sid,instructions=dict(en='Keep this photograph exactly as it is. Do not change its brightness, colours or framing.',zh='保持这张照片原样，不要改变明暗、色彩或构图。'),
    instruction_origin='AI_explicit_keep',kind='AI_new_explicit_keep',family='keep',results=[dict(candidate='executed',decision=decision,status='unchanged',image=str(path.relative_to(ROOT)),image_sha256=sha(path))],sheet=str(sheet.relative_to(ROOT)),sheet_sha256=sha(sheet),label_admitted=False)
   items.append(c)
  domain_index[source['domain']]+=1
 assert len(selection)==80 and len(items)==182 and Counter(family_counts.values())=={20:4}
 write(dest/'manifest.json',dict(at=stamp(),items=items,sheets=sheets,source_selection=selection,pair_family_groups=dict(family_counts),
  unique_photograph_groups=80,source_domains=dict(domain_index),new_AI_edit_cases=162,new_keep_cases=20,GPU_hours=0,editor=EDITOR_VERSION,
  policy='Actual new request/new execution/new direct visual review required. Original failed native requests are preserved, excluded, never silently weakened or counted as completed. No label is admitted here. Same original input for every paired task; no synthetic condition or user upload. Keep is exact pixel identity.'))
 print('Executed new requests:162edit20keep,80source groups40/40; admission pending')


if __name__=='__main__':main()

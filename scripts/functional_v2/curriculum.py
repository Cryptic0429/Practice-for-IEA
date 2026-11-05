"""Construct review candidates, never auto-admit labels from inverse parameters."""
import argparse,collections,json,math
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageOps
from functional_v1.curate import font
from functional_v1.editor import load_rgb,edit,save,statistics
from functional_v1.common import system_prompt
from .common import ROOT,DATA,EXP,PARENT,read,write,sha,stamp,EDITOR_VERSION

COND={
 'exposure':{'train':('请纠正照片曝光不足的问题，让整体光照恢复自然，保留原有颜色。','Correct the underexposure to restore natural overall lighting while retaining the original colours.'),
 'development':('这张照片曝光不足，请恢复合适的整体亮度，保持自然色彩。','This photograph is underexposed. Restore appropriate overall brightness and keep natural colours.'),
 'acceptance':('修复这张偏暗照片的曝光，使光照自然，保留色彩和细节。','Fix the exposure of this dark photo for natural lighting, preserving colours and detail.')},
 'contrast':{'train':('请修复灰蒙蒙的低对比度，使画面恢复自然的明暗层次。','Correct the washed-out low contrast to restore natural tonal separation.'),
 'development':('这张照片的明暗反差太平，请恢复自然的对比。','The tonal range looks too flat. Restore natural contrast.'),
 'acceptance':('画面发灰，缺少层次，请让明暗对比恢复正常。','The image looks washed out and lacks tonal separation. Bring the contrast back to normal.')},
 'saturation':{'train':('请修复颜色褪淡的问题，使色彩恢复自然，不要夸张增艳。','Correct the faded colours to restore natural saturation without exaggerated vibrancy.'),
 'development':('颜色显得褪淡，请恢复真实自然的色彩浓度。','The colours look faded. Restore realistic, natural colour intensity.'),
 'acceptance':('这张图颜色淡得不自然，请恢复正常的色彩，保持原来的明暗。','The colours in this image look unnaturally faded. Restore normal colour intensity while retaining the lighting.')}}
STYLE=[('white_balance','temperature',10,30),('saturation','saturation',12,40),('contrast','contrast',8,22),('brown_tone','sepia',35,95),('lighting','shadows',20,60)]

def style_text(tool,positive,strong,split):
    degree_zh='明显' if strong else '轻微';degree_en='noticeably' if strong else 'slightly'
    prefixes={'train':('请','Please '),'development':('我想让这张照片','For this photo, '),'acceptance':('把这张照片','Make this photograph ')}
    if tool=='sepia':
        zh=f'给照片添加{degree_zh}的棕褐色复古效果，保留明暗层次。'
        en=f'Give the photo a {"strong" if strong else "subtle"} brown sepia vintage look while retaining tonal detail.'
    else:
        terms={'temperature':('调暖整体色调' if positive else '调冷整体色调','warm the overall tone' if positive else 'cool the overall tone'),
            'saturation':('提高饱和度，让颜色更鲜艳' if positive else '降低饱和度，让颜色更淡雅','increase colour saturation' if positive else 'reduce colour saturation'),
            'contrast':('增强明暗对比' if positive else '减弱明暗反差','increase tonal contrast' if positive else 'soften tonal contrast'),
            'shadows':('提亮暗部' if positive else '压暗阴影','lift the shadows' if positive else 'darken the shadows')}
        z,e=terms[tool]
        zh=f'{prefixes[split][0]}{degree_zh}{z}，保留主体和细节，其他效果保持自然。'
        en=f'{"Please " if split=="train" else "For this photograph, "}{degree_en} {e} while preserving the subject and detail; keep other aspects natural.'
    return {'zh':zh,'en':en}


def main():
    assert not (DATA/'train-cases.json').exists(),'Existing candidates are immutable; revise explicitly'
    sources=read(DATA/'source-splits.json')['sources'];man=[]
    for split in ('train','development','acceptance'):
        cases=[]
        for domain in ('portrait','landscape'):
            domain_rows=[r for r in sources if r['split']==split and r['domain']==domain]
            for idx,r in enumerate(domain_rows):
                rgb=load_rgb(r['source_image']);scale=min(1,1024/max(rgb.shape[:2]))
                if scale<1:rgb=cv2.resize(rgb,(round(rgb.shape[1]*scale),round(rgb.shape[0]*scale)),interpolation=cv2.INTER_AREA)
                canonical=DATA/'inputs'/r['id']/'original.png';save(canonical,rgb)
                per=[];family=('exposure','contrast','saturation')[idx%3]
                levels=((20,65),(30,80),(40,100))[(idx//3)%3] if split=='train' else ((25,75) if split=='development' else (35,90))
                def add(name,inrgb,instructions,params,family_name,origin,expected,source_input=None,degradation=None):
                    caseid=f'{r["id"]}-{name}';ip=DATA/'inputs'/r['id']/f'{name}.png'
                    if source_input:ip=source_input
                    else:save(ip,inrgb)
                    tp=DATA/'targets'/f'{caseid}.png';result=edit(inrgb,params);save(tp,result)
                    case={'id':caseid,'group_id':r['group_id'],'split':split,'domain':domain,'source_id':r['id'],
                        'image':str(ip.relative_to(ROOT)),'input_sha256':sha(ip),'instructions':instructions,
                        'family':family_name,'action':name,'expected_change':expected,
                        'must_preserve':('保持面部结构及自然肤色，保留服装、背景和明暗细节。' if domain=='portrait' else '保持景物结构、天空层次、植被和水面自然色彩，保留明暗细节。'),
                        'failure_conditions':['未完成要求的方向或力度','过曝、暗部明显丢失、额外偏色或结构改动','复合要求有任何一项遗漏'],
                        'instruction_origin':origin,'english_origin':'AI-authored or AI translation, not a user trial request',
                        'original_source_image':r['source_image'],'original_source_sha256':r['source_sha256'],
                        'original_pool_request':r['original_request'],'canonical_photo':str(canonical.relative_to(ROOT)),
                        'target':str(tp.relative_to(ROOT)),'target_sha256':sha(tp),'parameters':params,
                        'degradation':degradation,'status':'unverified_candidate_not_training_data','editor':EDITOR_VERSION,
                        'target_input_policy':'Target/canonical photo/parameters/review never enter model input.'}
                    cases.append(case);per.append(case);return case
                conds=[]
                for k,level in enumerate(levels):
                    inverse=-level if family!='saturation' else round(100*(1/(1+level/100)-1),6)
                    degraded=edit(rgb,{family:inverse})
                    conds.append(add('restore-'+family+('-mild' if k==0 else '-strong'),degraded,dict(zip(('zh','en'),COND[family][split])),{family:level},
                        'lighting' if family=='exposure' else family,'AI-authored restoration request + known synthetic degradation, directly reviewed target required',
                        '修复可见的'+{'exposure':'曝光不足','contrast':'低对比灰雾','saturation':'色彩褪淡'}[family]+'，恢复自然观感；两种问题程度应采用不同力度。',degradation={family:inverse}))
                fam,tool,weak,strong=STYLE[idx%5];positive=tool=='sepia' or (idx//5)%2==0
                for k,value in enumerate((weak,strong)):
                    val=value if positive else (-round(value*.6) if tool=='shadows' else -value)
                    add('style-'+tool+('-weak' if k==0 else '-strong'),rgb,style_text(tool,positive,k==1,split),{tool:val},fam,
                        'AI-authored counterfactual strength request',
                        ('轻微' if k==0 else '明显')+'按指定方向调整'+tool+'；同图两档要求必须有可辨认的力度差异。',source_input=canonical)
                if split=='train' or idx%2==0:
                    cidx=idx%3
                    cp=[{'temperature':15,'saturation':-20},{'shadows':40,'contrast':10},{'highlights':-25,'temperature':-12}][cidx]
                    texts=[('整体暖一点，同时把颜色调得柔和、低饱和一些。','Warm the overall tone a little and make the colours softer and less saturated.'),
                        ('提亮暗部，同时轻微增强对比度，亮处不要过曝。','Lift the shadows and slightly increase contrast without blowing out the highlights.'),
                        ('把过亮的地方柔和压暗，同时增加一点冷色调。','Gently tone down the bright areas and add a little coolness.')][cidx]
                    add('compound',rgb,dict(zip(('zh','en'),texts)),cp,'compound','AI-authored two-clause request',
                        '同时完成两项要求；不能只执行第一项。',source_input=canonical)
                if split=='train' or idx%2==1:
                    ip=ROOT/conds[idx%2]['image'] if idx%3 else canonical;inp=load_rgb(ip)
                    add('identity',inp,{'zh':'请保持这张照片原样，不需要任何修图或自动增强。','en':'Keep this photograph exactly as supplied. Do not edit or automatically enhance it.'},{},'identity',
                        'AI-authored explicit no-op, including visibly degraded photos','按要求直接返回原图，像素保持原样。',source_input=ip)
                if split=='train':
                    # Every actual input and output appears; reduced sheets are not used for sharpening claims.
                    panels=[('Canonical source',rgb)]
                    for c in per:
                        if c['action'].startswith('restore-'):panels.extend([(c['action']+' input',load_rgb(ROOT/c['image'])),(c['action']+' target',load_rgb(ROOT/c['target']))])
                        elif c['action']=='identity':panels.append(('Identity INPUT = target',load_rgb(ROOT/c['image'])))
                        else:panels.append((c['action']+' target',load_rgb(ROOT/c['target'])))
                    canvas=Image.new('RGB',(2400,1600),'white');d=ImageDraw.Draw(canvas)
                    d.text((12,8),r['id']+' | '+domain+' | TRAINING TARGET REVIEW CANDIDATES',font=font(29),fill='black')
                    for i,(title,arr) in enumerate(panels):
                        x,y=(i%4)*600,60+(i//4)*510
                        d.text((x+10,y),title,font=font(21),fill='black')
                        im=ImageOps.contain(Image.fromarray(arr),(575,465),Image.Resampling.LANCZOS);canvas.paste(im,(x+10+(575-im.width)//2,y+32+(465-im.height)//2))
                    sheet=EXP/'label-review'/f'{r["id"]}.jpg';sheet.parent.mkdir(parents=True,exist_ok=True);canvas.save(sheet,quality=96,subsampling=0)
                    man.append({'source_id':r['id'],'sheet':str(sheet.relative_to(ROOT)),'sha256':sha(sheet),'case_ids':[c['id'] for c in per]})
        write(DATA/f'{split}-cases.json',cases)
        print(split,'cases',len(cases),'groups',len(domain_rows)*2)
    write(EXP/'label-review/manifest.json',man)
    write(EXP/'protocol.json',{'status':'cases_frozen_targets_pending_direct_visual_review','at':stamp(),
        'source_splits_sha256':sha(DATA/'source-splits.json'),'case_hashes':{s:sha(DATA/f'{s}-cases.json') for s in ('train','development','acceptance')},
        'parent':str(PARENT.relative_to(ROOT)),'parent_sha256':sha(PARENT/'adapter_model.safetensors'),
        'training':{'routes':['old-data matched continued-SFT control','new image-conditioned curriculum'],'epochs':4,'learning_rate':5e-5,'effective_batch':4,'seed':20261106,
            'quantization':'NF4 QLoRA','rank':8,'alpha':16,'layers':'language q/v only, frozen vision','rows_per_route':720},
        'selection':'Development visible complete + acceptable, both languages per semantic case, then severe damage count, then paired blind preference, then earliest epoch. L1 excluded.',
        'review':'Codex direct visual primary; original Qwen advisory only due round1 failed calibration. AI-assisted not user validation.',
        'evaluation_history':'Fresh source groups for round2; original historical val/test pools previously evaluated. No claim of unseen real-user photos.',
        'scope':'Five global task families plus compounds/no-op; controlled degradations and AI-authored instructions. Original natural-request candidates remain unadmitted.',
        'resource_limit':'Shared original 16 occupied GPU-hour ledger, GPUs0/2/3, protect1; stop at cap.',
        'input_allowlist':['input photo','ZH or EN request','same frozen v1.2 tool prompt'],
        'core_code_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/functional_v1'/f for f in ('common.py','editor.py','runtime.py')]}})
if __name__=='__main__':main()

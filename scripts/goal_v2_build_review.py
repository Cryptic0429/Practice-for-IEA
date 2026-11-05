"""Full-frame3x3 anonymous inspection sheets with separately saved source provenance."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
import cv2
from PIL import Image,ImageDraw,ImageFont,ImageOps
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def link_copy(source,destination):
    try:os.link(source,destination)
    except OSError:shutil.copyfile(source,destination)
    assert sha(source)==sha(destination)
def wrap(text,font,width):
    lines=[];line=''
    for word in ' '.join(text.split()).split(' '):
        proposed=(line+' '+word).strip()
        if font.getlength(proposed)>width and line:lines.append(line);line=word
        else:line=proposed
    if line:lines.append(line)
    return lines or ['No specific editing request supplied.']
def render(case,panels,source_dir,path,preview=False):
    small=ImageFont.load_default(size=18);title=ImageFont.load_default(size=24);caption=wrap('Instruction: '+case['instruction'],small,1740)
    top=65+24*len(caption)+20;cw,ch=600,450;image_h=395;canvas=Image.new('RGB',(1800,top+3*ch+65),'white');draw=ImageDraw.Draw(canvas)
    heading='VALIDATION LAYOUT PREVIEW' if preview else 'Anonymous independent-test review'
    draw.text((20,14),f'{heading} | Case {case["case"]:02d} | {case["id"]}',font=title,fill='black')
    for i,line in enumerate(caption):draw.text((20,55+24*i),line,font=small,fill='black')
    boxes=[]
    for index,panel in enumerate(panels):
        x=(index%3)*cw;y=top+(index//3)*ch;source=source_dir/panel['file'];assert sha(source)==panel['sha256']
        image=cv2.imread(str(source));assert image is not None;rgb=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));fitted=ImageOps.contain(rgb,(580,image_h),Image.Resampling.LANCZOS)
        draw.text((x+15,y+4),panel['label'],font=small,fill='black');bx=x+(cw-fitted.width)//2;by=y+35+(image_h-fitted.height)//2;canvas.paste(fitted,(bx,by));boxes.append([bx,by,bx+fitted.width,by+fitted.height])
        assert x<=bx<bx+fitted.width<=x+cw and y+35<=by<by+fitted.height<=y+35+image_h
    draw.text((20,top+3*ch+12),'Full frames; identical letterboxes; no colour adjustment or crop. Output order varies by case. AI-assisted review.',font=small,fill='black')
    canvas.save(path);return {'canvas':list(canvas.size),'image_boxes':boxes,'cell_size':[cw,ch],'crop':'none','colour_adjustment':'none','alignment':'fixed equal grid; every image contained within its cell'}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');a=ap.parse_args()
    lock=json.loads((OUT/'final-models-lock.json').read_text());names=lock['baseline_keys']+lock['candidate_keys'];assert len(names)==7
    plan=json.loads((OUT/'final-visual-plan.json').read_text())
    if a.preview:
        rows=json.loads((ROOT/'data/goal-v2-20261103/validation.json').read_text());row=rows[0];cases=[{'case':1,'id':row['id'],'group_id':row['group_id']}];lookup={row['id']:row};mapping={'1':{f'Output{i+1}':n for i,n in enumerate(names)}};role='validation_layout_preview';out=OUT/'final-review-layout-preview'
    else:
        protocol=json.loads((OUT/'final-protocol.json').read_text());assert sha(OUT/'final-visual-plan.json')==protocol['visual_plan_sha256'];assert sha(OUT/'final-visual-key.private.json')==plan['key_sha256']
        assert all(json.loads((OUT/'final-evaluation'/n/'status.json').read_text())['status']=='completed' for n in names)
        rows=json.loads(Path(protocol['test_source']['path']).read_text());lookup={r['id']:r for r in rows};cases=plan['cases'];mapping=json.loads((OUT/'final-visual-key.private.json').read_text())['case_mapping'];role='anonymous_final_review';out=OUT/'final-review'
    out.mkdir(exist_ok=False);(out/'panels').mkdir();(out/'cases').mkdir();predictions={}
    for name in names:
        if a.preview:
            vname=lock['models'][name].get('validation_name',name);directory=OUT/'evaluation'/f'validation-{vname}'
        else:directory=OUT/'final-evaluation'/name
        records=[json.loads(l) for l in (directory/'predictions.jsonl').open()];predictions[name]={r['id']:r for r in records}
    manifest={'role':role,'model_lock_sha256':sha(OUT/'final-models-lock.json'),'visual_plan_sha256':sha(OUT/'final-visual-plan.json'),'cases':[],'display_policy':'Native pixels decoded by same OpenCV pathway as evaluation; full-frame aspect-preserving equal letterboxes; no colour edits or crop.'};private=[]
    for spec in cases:
        row=lookup[spec['id']];user=next(m['content'] for m in row['conversations'] if m['role']=='user');instruction=row['extra_info']['instruction'];assert instruction in user
        case={'case':spec['case'],'id':row['id'],'group_id':row['group_id'],'instruction':instruction};directory=out/'cases'/f'case-{spec["case"]:02d}';directory.mkdir()
        sources=[('Input',Path(row['images'][0]),None),('Reference',Path(row['extra_info']['reference_image_path']),None)]
        for label,model in mapping[str(spec['case'])].items():
            rec=predictions[model][row['id']];path=Path(rec['scoring']['edited_image_path'])
            if not a.preview:assert sha(path)==rec['edited_image_sha256']
            sources.append((label,path,model))
        panels=[]
        for label,source,model in sources:
            file=label+source.suffix.lower();destination=directory/file;link_copy(source,destination);panels.append({'label':label,'file':file,'sha256':sha(destination)})
            private.append({'case':spec['case'],'panel':label,'model':model,'source':str(source),'source_sha256':sha(source),'alias':str(destination.relative_to(out))})
        geometry=render(case,panels,directory,out/'panels'/f'case-{spec["case"]:02d}.png',a.preview);case['panels']=panels;case['geometry']=geometry;manifest['cases'].append(case)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(OUT/(out.name+'-provenance.private.json')).write_text(json.dumps({'role':role,'sources':private},indent=2)+'\n');shutil.copyfile(__file__,out/'render-source.py')
    print(json.dumps({'role':role,'cases':len(cases),'panels_per_case':9,'output':str(out),'identities_not_printed':True}))
if __name__=='__main__':main()

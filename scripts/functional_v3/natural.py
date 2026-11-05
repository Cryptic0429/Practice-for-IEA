"""Original training-pool requests, re-labelled by direct visual review on native photos."""
import argparse,hashlib,json,textwrap,os
from .common import ROOT,ROUND2_DATA,EXP,DATA,PARENT,read,write,append,sha,stamp,GPUJob
SPECS={
 'train-013':('请只稍微提亮一点，并把色温往冷色方向调整。',[{'exposure':3,'temperature':-8},{'exposure':6,'temperature':-12},{'brightness':4,'temperature':-8}]),
 'train-030':('增强色彩，让颜色更鲜明。',[{'saturation':12},{'saturation':25},{'saturation':40}]),
 'train-035':('把这张照片提亮。',[{'exposure':5},{'exposure':10},{'shadows':25}]),
 'train-053':('添加棕褐色色调。',[{'sepia':35},{'sepia':65},{'sepia':95}]),
 'train-102':('把照片稍微调暗一点。',[{'exposure':-5},{'exposure':-10},{'brightness':-8}]),
 'train-039':('降低亮度。',[{'exposure':-8},{'exposure':-15},{'brightness':-12}]),
 'train-220':('提亮整张照片，突出树木、岩石和衣服的色彩。',[{'exposure':5,'saturation':10},{'exposure':10,'saturation':15},{'shadows':30,'saturation':15}]),
 'train-243':('把照片提亮。',[{'exposure':5},{'exposure':10},{'shadows':25}]),
 'train-273':('提亮照片并提高对比度。',[{'exposure':8,'contrast':6},{'exposure':15,'contrast':10},{'shadows':30,'contrast':8}]),
 'train-278':('请大幅提高照片的明亮程度。',[{'exposure':15},{'exposure':25},{'shadows':55,'exposure':8}]),
}

def prepare():
    from functional_v1.editor import load_rgb,edit,save
    if (DATA/'natural-cases.json').exists():raise RuntimeError('Frozen cases already exist')
    sources={c['source_id']:c for c in read(ROUND2_DATA/'train-final-cases.json')}
    cases=[]
    for sid,(zh,params) in SPECS.items():
        src=sources[sid];cid='natural-'+sid
        assert src['split']=='train'
        image=src['canonical_photo'];proposals=[]
        for i,parameters in enumerate(params,1):
            file=DATA/'natural-proposals'/f'{cid}-{i}.png';save(file,edit(load_rgb(ROOT/image),parameters))
            proposals.append({'id':f'proposal-{i}','parameters':parameters,'image':str(file.relative_to(ROOT)),'sha256':sha(file)})
        cases.append({'id':cid,'source_id':sid,'domain':src['domain'],'group_id':src['group_id'],'split':'train',
          'image':image,'input_sha256':sha(ROOT/image),'instructions':{'zh':zh,'en':src['original_pool_request']},
          'family':'original-natural-request','action':'original-natural-request',
          'instruction_origin':'Untouched original GIER English request; AI Chinese translation; not actual present user feedback.',
          'expected_change':'完整完成原始请求的方向、力度及所有组成；以可见图片判断，不能只调用正确工具。',
          'must_preserve':'主体结构、未要求的颜色和明暗层次；避免过曝、暗部损失和非预期肤色变化。',
          'proposals':proposals})
    write(DATA/'natural-cases.json',cases)
    protocol=read(EXP/'collection-protocol.json');protocol['native_natural_addition']={'at':stamp(),'cases_sha256':sha(DATA/'natural-cases.json'),'cases':10,'photo_groups':10,'domains':{'portrait':5,'landscape':5},'original_english_requests':10,'AI_translations':10,'admission':'Re-review request/output for every actual natural target; no inherited reference from a rewritten task. Use training groups only, before any natural parent inference.'};write(EXP/'collection-protocol.json',protocol)

def infer(gpu):
    cases=read(DATA/'natural-cases.json');protocol=read(EXP/'collection-protocol.json')
    assert sha(DATA/'natural-cases.json')==protocol['native_natural_addition']['cases_sha256']
    assert sha(PARENT/'adapter_model.safetensors')==protocol['parent_sha256']
    out=EXP/'natural-collection';out.mkdir(exist_ok=True);file=out/'predictions.jsonl'
    rows=list(map(json.loads,file.read_text().splitlines())) if file.exists() else [];done={(r['case_id'],r['language']) for r in rows}
    if len(done)==20:return
    with GPUJob(gpu,'v3-collect-natural-training-requests',max_seconds=1800) as budget:
        import torch
        from functional_v1.runtime import load_model,processor,predict
        from functional_v1.editor import load_rgb,edit,save
        torch.set_num_threads(4);model=load_model(PARENT);p=processor()
        for c in cases:
            for lang in ('zh','en'):
                if (c['id'],lang) in done:continue
                budget.tick();assert c['split']=='train' and c['source_id'].startswith('train-')
                row={'case_id':c['id'],'language':lang,'input_image':c['image'],'instruction':c['instructions'][lang],'status':'failed','parent_sha256':protocol['parent_sha256']}
                try:
                    response,parameters=predict(model,p,c,lang);output=out/'images'/f'{c["id"]}-{lang}.png';save(output,edit(load_rgb(ROOT/c['image']),parameters))
                    row.update(status='executed',parameters=parameters,response=response,output_image=str(output.relative_to(ROOT)),output_sha256=sha(output))
                except Exception as e:row['error']=f'{type(e).__name__}: {e}'
                row['at']=stamp();append(file,row);rows.append(row);done.add((c['id'],lang))
        write(out/'status.json',{'status':'completed','predictions':len(rows),'at':stamp()})

def render():
    from PIL import Image,ImageDraw,ImageOps
    from functional_v1.curate import font
    out=EXP/'natural-review';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Review bundle already exists')
    preds=list(map(json.loads,(EXP/'natural-collection/predictions.jsonl').read_text().splitlines()));manifest=[]
    for c in read(DATA/'natural-cases.json'):
        unique={x['sha256']:{'id':x['id'],'image':x['image'],'parameters':x['parameters'],'sha256':x['sha256']} for x in c['proposals']}
        for r in preds:
            if r['case_id']==c['id'] and r['status']=='executed':unique.setdefault(r['output_sha256'],{'id':'policy-'+r['language'],'image':r['output_image'],'parameters':r['parameters'],'sha256':r['output_sha256']})
        entries=sorted(unique.values(),key=lambda x:hashlib.sha256((c['id']+x['sha256']).encode()).hexdigest())
        canvas=Image.new('RGB',(1800,1370),'white');d=ImageDraw.Draw(canvas)
        for j,line in enumerate(textwrap.wrap(c['instructions']['en'],115)):d.text((12,12+j*30),line,fill='black',font=font(24))
        for i,(label,path) in enumerate([('INPUT',c['image'])]+[(f'Output {i+1}',e['image']) for i,e in enumerate(entries)]):
            x,y=(i%3)*600,130+(i//3)*610;d.text((x+12,y),label,fill='black',font=font(24));image=ImageOps.contain(Image.open(ROOT/path).convert('RGB'),(580,550));canvas.paste(image,(x+10+(580-image.width)//2,y+40+(550-image.height)//2))
        file=out/(c['id']+'.jpg');canvas.save(file,quality=96,subsampling=0)
        manifest.append({'case_id':c['id'],'sheet':str(file.relative_to(ROOT)),'sha256':sha(file),'instructions':c['instructions'],'entries':entries})
    write(out/'manifest.json',manifest)
    print(json.dumps({'sheets':len(manifest),'request_origin':'10 untouched original EN requests on existing training groups; ZH translations, not independent cases'}))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','infer','render']);ap.add_argument('--gpu',choices=['0','2','3']);a=ap.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='infer':infer(a.gpu)
    else:render()
if __name__=='__main__':main()

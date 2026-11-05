"""CPU-only frozen-output replay. No training, fitting, parser repair or holdout opening."""
import json, math, textwrap
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from functional_v1.common import ROOT,read,write,sha,stamp,TOOLS
from functional_v1.editor import load_rgb,edit as legacy,save,statistics
from functional_v1.curate import font
from functional_v6.evaluate import EVAL,case_file
from .editor import edit,tonal_value

EXP=ROOT/'experiments/functional-v7-20261105'
MODELS=('current-default','continued-e4')
VARIANTS=('bounded-fixed','bounded-adaptive')
IDS={'native-development':['native-dev-001','native-dev-008','native-dev-011','native-dev-070','native-dev-108'],
     'old-native':['natural-019','natural-028','natural-030']}

def test():
    # Scope: mathematical execution invariants, not practical image quality.
    x=np.linspace(0,1,65537,dtype=np.float32)
    rgb=np.stack(np.meshgrid(np.arange(256,dtype=np.uint8),np.arange(256,dtype=np.uint8)),axis=-1)
    rgb=np.concatenate([rgb,((rgb[...,0].astype(int)+rgb[...,1])/2).astype(np.uint8)[...,None]],axis=-1)
    checks=[]
    for variant in VARIANTS:
        assert np.array_equal(edit(rgb,{},variant),rgb)
        assert np.array_equal(edit(rgb,dict.fromkeys(TOOLS,0),variant),rgb)
        for tool in ('exposure','brightness','contrast'):
            for v in (-100,-50,-10,0,10,50,100):
                y=tonal_value(x,{tool:v},variant)
                assert np.isfinite(y).all() and y.min()>=0 and y.max()<=1
                assert np.diff(y).min()>=-1e-7 and y[0]==0 and y[-1]==1
                if tool!='contrast':assert ((y-x)*v>=-1e-7).all()
                if tool=='contrast':
                    lo,hi=np.searchsorted(x,[.2,.8]);delta=(y[hi]-y[lo])-(x[hi]-x[lo]);assert delta*v>=-1e-7
        for tool in TOOLS:
            if tool in ('exposure','brightness','contrast'):continue
            for v in (0,50,100) if tool=='sepia' else (-100,-25,0,25,100):
                # Every untouched colour / range tool stays byte-identical.
                assert np.array_equal(edit(rgb,{tool:v},variant),legacy(rgb,{tool:v}))
        for bad in ({'sepia':-1},{'exposure':101},{'sharpen':2},{'brightness':True},{'contrast':float('nan')}):
            try:edit(rgb,bad,variant)
            except ValueError:pass
            else:raise AssertionError('invalid parameter accepted')
        checks.append(dict(variant=variant,neutral_pixel_identical=True,tone_curves_bounded_monotone=True,
            signed_directions_verified=True,untouched_tools_byte_identical=True,illegal_parameters_still_rejected=True))
    return checks

def main():
    # Protocol written before rendering ANY development result with new tools.
    lock=EXP/'editor-diagnostic-protocol.json';assert not lock.exists(),'Preserve completed diagnostic; use a new version.'
    checks=test();write(EXP/'editor-invariant-tests.json',dict(at=stamp(),status='passed',checks=checks))
    sources=[]
    for split,ids in IDS.items():
        file=case_file(split)
        for mid in MODELS:
            p=EVAL/split/mid/'predictions.jsonl'
            sources.append(dict(split=split,model_id=mid,cases_sha256=sha(file),predictions=str(p.relative_to(ROOT)),predictions_sha256=sha(p)))
    write(lock,dict(at=stamp(),stage='targeted_development_diagnostic_not_validation',models=MODELS,variants=VARIANTS,
        cases=IDS,sources=sources,code_hashes={str(Path(f).relative_to(ROOT)):sha(f) for f in (Path(__file__),Path(__file__).with_name('editor.py'),ROOT/'scripts/functional_v1/editor.py')},
        purpose='Separate legacy early clipping / contrast pivot from learned parameter errors. No model output changes, parameter fitting or training.',
        review='Direct Codex visual review of anonymous same-scale full frames; source parameters and renderer identities hidden until notes saved. Final counts cannot prove practical readiness.',
        declared_limits=['Equal slider values do not guarantee equal visual strength with different transfer curves.',
          'The eight cases are a targeted, already-seen development diagnostic, not a balanced or independent test.',
          'New rendering does not repair wrong direction, missing tools, illegal JSON or denoise compressed images.',
          'All existing results, live model/editor, training targets and sealed acceptance remain unchanged.'],
        literature=[dict(url='https://openaccess.thecvf.com/content_CVPR_2020/html/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.html',
            takeaway='Bounded monotonic tone curves are an established design principle; this diagnostic uses its own rational/logit global curves, not Zero-DCE architecture or weights.'),
          dict(url='https://arxiv.org/abs/2009.14468',takeaway='Image-adaptive LUTs motivate separating tone mapping capacity from language prediction. No LUT learning is implemented here.')]))
    records=[];unique={};mapping=[];sheets=[]
    for split,ids in IDS.items():
        cases={c['id']:c for c in read(case_file(split))}
        for cid in ids:
            c=cases[cid];before=load_rgb(ROOT/c['image']);assert sha(ROOT/c['image'])==c['input_sha256']
            entries={}
            for mid in MODELS:
                rows=[r for r in map(json.loads,(EVAL/split/mid/'predictions.jsonl').read_text().splitlines()) if r['case_id']==cid]
                assert len(rows)==2
                for r in rows:
                    if r['status']!='executed':
                        records.append(dict(split=split,model_id=mid,case_id=cid,language=r['language'],status='failed',error=r.get('error')));continue
                    old=legacy(before,r['parameters']);assert np.array_equal(old,load_rgb(ROOT/r['output_image']))
                    for variant in ('legacy',)+VARIANTS:
                        out=old if variant=='legacy' else edit(before,r['parameters'],variant)
                        digest=__import__('hashlib').sha256(out.tobytes()).hexdigest()
                        key=cid+'-'+digest[:12]
                        path=EXP/'editor-diagnostic/images'/f'{key}.png'
                        if key not in unique:
                            save(path,out);unique[key]=dict(item_id=key,case_id=cid,image=str(path.relative_to(ROOT)),image_sha256=sha(path),pixel_sha256=digest)
                        entries[key]=unique[key]
                        row=dict(split=split,model_id=mid,case_id=cid,language=r['language'],variant=variant,status='executed',
                            parameters=r['parameters'],item_id=key,input_statistics=statistics(before),output_statistics=statistics(out))
                        records.append(row);mapping.append({k:row[k] for k in ('split','model_id','case_id','language','variant','item_id')})
            items=sorted(entries.values(),key=lambda r:r['item_id'])
            for offset in range(0,len(items),5):
                subset=items[offset:offset+5];columns=min(3,1+len(subset));height=120+600*math.ceil((1+len(subset))/columns)
                canvas=Image.new('RGB',(columns*600,height),'white');d=ImageDraw.Draw(canvas)
                d.text((16,10),cid,fill='black',font=font(24))
                for j,line in enumerate(textwrap.wrap(c['instructions']['en'],43*columns)):d.text((16,48+26*j),line,fill='black',font=font(22))
                for i,(label,path) in enumerate([('INPUT',ROOT/c['image'])]+[(r['item_id'][-12:],ROOT/r['image']) for r in subset]):
                    x,y=(i%columns)*600,120+(i//columns)*600;d.text((x+12,y),label,fill='black',font=font(22))
                    with Image.open(path) as im:fitted=ImageOps.contain(im.convert('RGB'),(580,550),Image.Resampling.LANCZOS)
                    canvas.paste(fitted,(x+10+(580-fitted.width)//2,y+40+(550-fitted.height)//2))
                path=EXP/'editor-diagnostic'/f'{cid}-{offset//5+1:02d}.jpg';canvas.save(path,quality=95,subsampling=0)
                sheets.append(dict(case_id=cid,sheet=str(path.relative_to(ROOT)),sha256=sha(path),items=[r['item_id'] for r in subset]))
    write(EXP/'editor-diagnostic/manifest.json',dict(at=stamp(),protocol_sha256=sha(lock),items=list(unique.values()),mapping=mapping,sheets=sheets))
    write(EXP/'editor-diagnostic/replays.json',records)
    print(json.dumps(dict(unique_images=len(unique),sheets=len(sheets),language_renderer_records=len(records),GPU_hours=0)))

if __name__=='__main__':main()

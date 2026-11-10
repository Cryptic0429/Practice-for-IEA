"""Anonymous executed-pixel review, with explicit no-image decisions and crops."""
import hashlib
import json
import os
import textwrap
from PIL import Image,ImageDraw,ImageOps
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.curate import font
from functional_v15.native_labels import CRITICAL_REGIONS
from .audit_resolution import EXP


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    result=read(EXP/'result.json');protocol=read(EXP/'protocol.json')
    out=EXP/'review';assert not out.exists();out.mkdir()
    items=[]
    for case in protocol['cases']:
        cid=case['id'];preds={r['image_max_pixels']:r for r in result['records'] if r['case_id']==cid}
        order=[262144,1048576] if int(hashlib.sha256(cid.encode()).hexdigest()[:2],16)%2 else [1048576,262144]
        canvas=Image.new('RGB',(1800,820),'white');draw=ImageDraw.Draw(canvas)
        draw.text((12,10),cid+' | ANONYMOUS TRAIN-ONLY DIAGNOSTIC',font=font(22),fill='black')
        for n,line in enumerate(textwrap.wrap(case['instructions']['en'],120)):
            draw.text((12,48+n*25),line,font=font(20),fill='black')
        assets=[]
        for index,(label,cap) in enumerate([('INPUT',None),('A',order[0]),('B',order[1])]):
            draw.text((index*600+12,140),label,font=font(22),fill='black')
            path=ROOT/case['image'] if cap is None else (ROOT/preds[cap]['output_image'] if preds[cap]['output_image'] else None)
            if path is None:
                draw.text((index*600+35,390),'NO RESULT: '+preds[cap]['status'],font=font(24),fill='black')
                continue
            expected=case['input_sha256'] if cap is None else preds[cap]['output_sha256'];assert sha(path)==expected
            with Image.open(path) as im:
                rgb=im.convert('RGB');w,h=rgb.size
                fitted=ImageOps.contain(rgb,(580,610),Image.Resampling.LANCZOS)
                canvas.paste(fitted,(index*600+10+(580-fitted.width)//2,180+(610-fitted.height)//2))
                regions=[('whole',[0,0,w,h])] if max(w,h)<=1600 else CRITICAL_REGIONS[cid]
                for name,box in regions:
                    crop=rgb.crop(box);target=out/f'{cid}-{label}-{name}.png';crop.save(target,compress_level=1)
                    with Image.open(target) as check:assert check.tobytes()==crop.tobytes()
                    assets.append(dict(label=label,region=name,path=str(target.relative_to(ROOT)),sha256=sha(target),
                        source=str(path.relative_to(ROOT)),source_sha256=expected,source_size=[w,h],box=box))
        path=out/f'{cid}.jpg';canvas.save(path,quality=96,subsampling=0)
        items.append(dict(case_id=cid,instructions=case['instructions'],sheet=str(path.relative_to(ROOT)),sheet_sha256=sha(path),
            native=assets,anonymous_mapping=dict(A=order[0],B=order[1]),
            decisions={label:preds[cap]['status'] for label,cap in [('A',order[0]),('B',order[1])]},
            outputs_identical=preds[262144]['output_sha256']==preds[1048576]['output_sha256']))
    write(out/'manifest.json',dict(at=stamp(),items=items,protocol_sha256=sha(EXP/'protocol.json'),
        scope='TRAIN-only reused sources; anonymous direct review. Decline is no result, never input fallback. Native integer crops unscaled; large-photo quality limited to listed regions. No model training or heldout claim.'))
    print('Prepared4 TRAIN-only comparisons with native-region exports.')


if __name__=='__main__':main()

"""Preserve first retrieval and add further visually triaged real scenery."""
from collections import Counter
import json,textwrap
from PIL import Image,ImageOps,ImageDraw
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.curate import font
from .source_candidates import EXP,OLD


def main():
    first=EXP/'source-review/manifest.json';old=read(first);out=EXP/'source-review-extended'
    assert not (out/'manifest.json').exists()
    bank={c['id']:c for c in read(OLD/'source-extension/train/manifest.json')['candidates']}
    used={c['group_id'] for c in read(ROOT/'data/training/functional-v11-20261106/cases.json')}
    seen=used|{c['group_id'] for c in old['candidates']}
    proofs={str(p.relative_to(ROOT)):sha(p) for p in sorted((OLD/'source-extension/train').glob('overview-0[5-7].jpg'))}
    rows=[]
    for number in [339,348,372,505,506,511,519,521,526,529,533,538,551,552,446,443]:
        c=bank[f'train-extension-{number:03d}']
        if c['group_id'] in seen:continue
        assert c['split']=='train' and sha(c['source_image'])==c['source_sha256']
        seen.add(c['group_id']);rows.append(dict(c,domain='landscape',domain_evidence_kind='new_overview_domain_triage_only',overview_proofs=proofs))
    sheets=[]
    for start in range(0,len(rows),8):
        subset=rows[start:start+8];canvas=Image.new('RGB',(1800,1220),'white');d=ImageDraw.Draw(canvas)
        d.text((12,10),'Additional landscape source/request candidates, no target admitted',font=font(24),fill='black')
        for i,c in enumerate(subset):
            x,y=i%4*450,55+i//4*580;d.text((x+8,y),c['id'],font=font(20),fill='black')
            for j,line in enumerate(textwrap.wrap(c['instruction'],42)[:4]):d.text((x+8,y+28+22*j),line,font=font(17),fill='black')
            im=ImageOps.contain(ImageOps.exif_transpose(Image.open(c['source_image'])).convert('RGB'),(430,420));canvas.paste(im,(x+10+(430-im.width)//2,y+130+(420-im.height)//2))
        p=out/f'landscape-extra-{start//8+1:02d}.jpg';p.parent.mkdir(parents=True,exist_ok=True);canvas.save(p,quality=96,subsampling=0);sheets.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),ids=[c['id'] for c in subset]))
    combined=old['candidates']+rows
    write(out/'manifest.json',dict(old,at=stamp(),candidates=combined,sheets=old['sheets']+sheets,
        counts=dict(Counter(c['domain'] for c in combined)),extension=dict(previous_manifest=str(first.relative_to(ROOT)),previous_sha256=sha(first),added=len(rows),reason='Only32 eligible new scenery candidates in first retrieval; expand actual source review before training, no label-count shortcut.')))
    print('Extended candidates',dict(Counter(c['domain'] for c in combined)))


if __name__=='__main__':main()

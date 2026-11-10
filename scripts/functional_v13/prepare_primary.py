"""Execute source-specific primary proposals, preserving all candidate pixels."""
import json,textwrap
from collections import Counter
from PIL import Image,ImageDraw,ImageOps
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb,save
from functional_v1.curate import font
from functional_v11.outcomes import execute,validate_decision,EDITOR_VERSION
from .source_candidates import EXP


def main():
    dest=EXP/'primary-candidates';assert not (dest/'manifest.json').exists()
    source_manifest=EXP/'source-review-extended/manifest.json';m=read(source_manifest)
    bank={c['id']:c for c in m['candidates']}
    obs=EXP/'source-observations.jsonl';notes=list(map(json.loads,obs.read_text().splitlines()))
    assert len(notes)==len(bank)==97 and len({n['id'] for n in notes})==97
    amendments=EXP/'source-amendments.jsonl'
    changes={a['id']:a for a in map(json.loads,amendments.read_text().splitlines())}
    isolation=read(EXP/'source-isolation-extended.json');eligible=set(isolation['remaining_ids'])
    items=[]
    for raw in notes:
        if not raw['retain']:continue
        n={**raw,**changes.get(raw['id'],{})};c=bank[n['id']]
        assert n['id'] in eligible and sha(c['source_image'])==c['source_sha256']
        rgb=load_rgb(c['source_image']);out=dest/n['id'];out.mkdir(parents=True,exist_ok=True)
        source=out/'input.png'
        if source.exists():
            import numpy as np
            assert np.array_equal(load_rgb(source),rgb)
        else:save(source,rgb)
        results=[]
        for label,multiplier in [('A',1.),('softer',.75),('stronger',1.25)]:
            action=n.get('action','edit')
            if action=='edit':
                params={k:max(0 if k=='sepia' else -100,min(100,round(v*multiplier))) for k,v in n['parameters'].items()}
                params={k:v for k,v in params.items() if v}
                decision=dict(action='edit',parameters=params)
            else:decision=dict(action=action,**({'reason':n['reason']} if action=='decline' else {}))
            validate_decision(decision);result=execute(rgb,decision);target=None
            if result['status']!='declined':target=out/(label+'.png');save(target,result['result'])
            else:assert result['result'] is None
            results.append(dict(candidate=label,decision=decision,status=result['status'],
                image=str(target.relative_to(ROOT)) if target else None,image_sha256=sha(target) if target else None))
            if action!='edit':break
        canvas=Image.new('RGB',(1760,820),'white');d=ImageDraw.Draw(canvas)
        d.text((12,10),n['id']+' | '+n['primary_origin'],font=font(22),fill='black')
        for j,line in enumerate(textwrap.wrap(n['en'],135)[:3]):d.text((12,42+24*j),line,font=font(19),fill='black')
        pictures=[('INPUT',source)]+[(r['candidate'],ROOT/r['image'] if r['image'] else None) for r in results]
        for i,(label,path) in enumerate(pictures):
            x=i*440;d.text((x+10,115),label,font=font(21),fill='black')
            if path:
                im=ImageOps.contain(Image.open(path).convert('RGB'),(420,650));canvas.paste(im,(x+10+(420-im.width)//2,155+(650-im.height)//2))
            else:d.text((x+10,440),'DECLINED: '+n['reason'],font=font(17),fill='black')
        sheet=out/'review.jpg';canvas.save(sheet,quality=96,subsampling=0)
        items.append(dict(id=n['id'],domain=n['domain'],group_id=n['group_id'],source_image=c['source_image'],source_sha256=c['source_sha256'],
            input_image=str(source.relative_to(ROOT)),input_sha256=sha(source),instructions=dict(en=n['en'],zh=n['zh']),
            instruction_origin=n['primary_origin'],original_instruction=c['instruction'],native_annotation=c['native_annotation'],
            source_note=n,results=results,sheet=str(sheet.relative_to(ROOT)),sheet_sha256=sha(sheet),label_admitted=False))
    write(dest/'manifest.json',dict(at=stamp(),items=items,counts=dict(Counter(r['domain'] for r in items)),
        executed_edit_candidates=sum(r['status']!='declined' for i in items for r in i['results']),
        source_manifest_sha256=sha(source_manifest),source_observations_sha256=sha(obs),source_amendments_sha256=sha(amendments),
        source_isolation_sha256=sha(EXP/'source-isolation-extended.json'),editor=EDITOR_VERSION,GPU_hours=0,
        policy='All candidates actually executed. Contact sheets show entire images with identical letterboxes and no colour adjustment. No L1 or automatic quality choice. Final labels require direct visual review, including original-size risk followups; counts are proposals, not actual training data.'))
    print('Executed primary proposals:',dict(Counter(i['domain'] for i in items)))


if __name__=='__main__':main()

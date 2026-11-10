"""Prepare traceable training-source review sheets, not training labels."""
import json
import hashlib
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.curate import font

EXP = ROOT/'experiments/functional-v13-20261106'
OLD = ROOT/'experiments/functional-v6-20261105'


def main():
    out = EXP/'source-review'
    assert not (out/'manifest.json').exists()
    used = {c['group_id'] for c in read(ROOT/'data/training/functional-v11-20261106/cases.json')}
    banks, notes = {}, {}
    provenance = {}
    for folder in (OLD/'source-review', OLD/'source-extension/train'):
        mp,np=folder/'manifest.json',folder/'observations.jsonl'
        provenance[str(mp.relative_to(ROOT))]=sha(mp)
        provenance[str(np.relative_to(ROOT))]=sha(np)
        for c in read(mp)['candidates']:banks[c['id']]=c
        for n in map(json.loads,np.read_text().splitlines()):notes[n['id']]=n
    # These additional scene candidates were visually triaged from the existing
    # first four overviews. This does not approve scope, detail or edit labels.
    additional=[20,30,35,38,53,71,81,92,113,137,160,161,183,186,191,194,196,202,205,206,211,231,246,250,251,262,263,294]
    overview_proofs={str(p.relative_to(ROOT)):sha(p) for p in
        sorted((OLD/'source-extension/train').glob('overview-0[1-4].jpg'))}
    candidates=[];seen=set(used)
    # User-shown case12 remains a disclosed regression, never newly admitted.
    seen.add(banks['train-native-168']['group_id'])
    for domain in ('portrait','landscape'):
        rows=[]
        for sid,n in notes.items():
            c=banks[sid]
            if c['split']!='train' or not n['source_usable'] or n['domain']!=domain:continue
            if domain=='portrait' and n['request_scope']!='global':continue
            if sid in ('train-extension-049','train-extension-352','train-extension-357'):continue
            assert sha(ROOT/n['sheet'])==n['sheet_sha256']
            rows.append(dict(c,domain=domain,domain_evidence=n,domain_evidence_kind='inherited_direct_AI_source_classification'))
        if domain=='landscape':
            for num in additional:
                c=banks[f'train-extension-{num:03d}']
                rows.append(dict(c,domain=domain,domain_evidence_kind='new_overview_domain_triage_only',overview_proofs=overview_proofs))
        rows.sort(key=lambda r:hashlib.sha256(('functional-v13-source-candidates:'+r['id']).encode()).hexdigest())
        kept=0
        for c in rows:
            if c['group_id'] in seen:continue
            assert c['split']=='train' and sha(c['source_image'])==c['source_sha256']
            seen.add(c['group_id']);candidates.append(c);kept+=1
            if kept==50:break
    sheets=[]
    for domain in ('portrait','landscape'):
        rows=[c for c in candidates if c['domain']==domain]
        for start in range(0,len(rows),8):
            subset=rows[start:start+8];canvas=Image.new('RGB',(1800,1220),'white');d=ImageDraw.Draw(canvas)
            d.text((12,10),domain+' | source/request review only; no admitted label',font=font(24),fill='black')
            for i,c in enumerate(subset):
                x,y=i%4*450,55+i//4*580
                d.text((x+8,y),c['id'],font=font(20),fill='black')
                import textwrap
                for j,line in enumerate(textwrap.wrap(c['instruction'],42)[:4]):d.text((x+8,y+28+22*j),line,font=font(17),fill='black')
                im=ImageOps.contain(ImageOps.exif_transpose(Image.open(c['source_image'])).convert('RGB'),(430,420))
                canvas.paste(im,(x+10+(430-im.width)//2,y+130+(420-im.height)//2))
            target=out/f'{domain}-{start//8+1:02d}.jpg';target.parent.mkdir(parents=True,exist_ok=True);canvas.save(target,quality=96,subsampling=0)
            sheets.append(dict(path=str(target.relative_to(ROOT)),sha256=sha(target),ids=[c['id'] for c in subset]))
    write(out/'manifest.json',dict(at=stamp(),candidates=candidates,sheets=sheets,source_files=provenance,
        counts=dict(Counter(c['domain'] for c in candidates)),parent_unique_original_groups=len(used),
        policy='Original training-pool source candidates only, not label admission. Existing domain notes inherited; new overview domain triage must be checked on detail sheets. Original local/ambiguous text is not admitted as global supervision. No target, outcome or acceptance output viewed; no training started.',
        known_case12_excluded=True,labels_admitted=False))
    print('Source candidates:',dict(Counter(c['domain'] for c in candidates)))


if __name__=='__main__':main()

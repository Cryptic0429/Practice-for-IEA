"""Challenge each inherited quality refusal with a matched shadow-bearing attempt."""
import os,textwrap
from collections import defaultdict
from PIL import Image,ImageOps,ImageDraw
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb,save
from functional_v1.curate import font
from functional_v11.editor import edit as old_edit
from .editor import edit as new_edit

EXP=ROOT/'experiments/functional-v15-20261106'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    target=EXP/'quality-challenges/manifest.json';assert not target.exists()
    source=ROOT/'data/training/functional-v13-20261106/cases.json'
    cases=[c for c in read(source) if c['action']=='decline' and c['decision']['reason']=='unreliable_quality'];assert len(cases)==44
    previous=ROOT/'experiments/functional-v11-20261106/quality-revalidation/manifest.json';bank=defaultdict(list)
    for row in read(previous)['attempts']:bank[row['case_id']].append(row)
    assert all(bank[c['id']] for c in cases)
    rows=[]
    for c in cases:
        old=next((r for r in bank[c['id']] if r['parameters'].get('shadows',0)),bank[c['id']][0])
        assert old['instructions']==c['instructions'] and old['input']['sha256']==c['input_sha256']
        parameters=dict(old['parameters']);parameters['shadows']=max(50,parameters.get('shadows',0))
        # Both tools receive exactly the same challenging parameters. This is
        # not a new reliable label and does not imply the old attempt failed.
        rgb=load_rgb(ROOT/c['image']);assert sha(ROOT/c['image'])==c['input_sha256']
        folder=target.parent/c['id'];folder.mkdir(parents=True,exist_ok=False)
        sources=[('INPUT',ROOT/c['image'])]
        for label,function in [('OLD TOOL, SAME CHALLENGE',old_edit),('SHARED SHADOW GAIN CHALLENGE',new_edit)]:
            out=folder/('old.png' if label.startswith('OLD') else 'candidate.png');save(out,function(rgb,parameters,'bounded-fixed'));sources.append((label,out))
        canvas=Image.new('RGB',(1800,820),'white');draw=ImageDraw.Draw(canvas)
        draw.text((16,12),c['id']+' | QUALITY REFUSAL CHALLENGE, NOT ADMITTED',font=font(23),fill='black')
        for n,line in enumerate(textwrap.wrap(c['instructions']['en'],140)):
            draw.text((16,48+n*25),line,font=font(20),fill='black')
        for n,(label,path) in enumerate(sources):
            draw.text((n*600+12,150),label,font=font(19),fill='black')
            with Image.open(path) as im:image=ImageOps.contain(im.convert('RGB'),(580,600),Image.Resampling.LANCZOS)
            canvas.paste(image,(n*600+10+(580-image.width)//2,190+(600-image.height)//2))
        panel=folder/'review.jpg';canvas.save(panel,quality=95,subsampling=0)
        rows.append(dict(id=c['id'],domain=c['domain'],family=c['family'],group_id=c['group_id'],instructions=c['instructions'],
            instruction_origin=c['instruction_origin'],input=dict(path=c['image'],sha256=c['input_sha256'],size=list(rgb.shape[1::-1])),
            previous_attempt_source=dict(path=str(previous.relative_to(ROOT)),sha256=sha(previous),id=old['id'],parameters=old['parameters']),
            parameters=parameters,challenge_rule='First prior shadow-bearing attempt, otherwise first prior attempt; shadows raised to at least50, other values unchanged. Both tools actually execute identical new parameters.',
            old=dict(path=str(sources[1][1].relative_to(ROOT)),sha256=sha(sources[1][1])),candidate=dict(path=str(sources[2][1].relative_to(ROOT)),sha256=sha(sources[2][1])),
            panel=dict(path=str(panel.relative_to(ROOT)),sha256=sha(panel)),status='not_viewed_not_admitted',
            no_impossibility_claim=True,label_admitted=False))
    write(target,dict(at=stamp(),source_cases_sha256=sha(source),previous_attempts_sha256=sha(previous),items=rows,GPU_hours=0,new_labels=0,
        rule='Direct AI functional/quality review required. A successful attempt can replace a refusal label; failed finite attempts do not prove all possible parameters impossible. All training-pool only; no development/acceptance inputs.'))
    print('44 quality-refusal challenge pairs generated, no grades or label admission.')


if __name__=='__main__':main()

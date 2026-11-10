"""Preserve full current-input pixels beside the selected edit/refusal witness."""
import json
from PIL import Image,ImageDraw
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.curate import font
from .prepare_quality_data import EXP


def main():
    source=EXP/'quality-candidates/manifest.json';manifest=read(source)
    notes_path=EXP/'quality-triage-notes.jsonl'
    notes={r['id']:r for r in map(json.loads,notes_path.read_text().splitlines())}
    assert len(notes)==len(manifest['items'])==64
    target=EXP/'quality-native-review/manifest.json';assert not target.exists()
    rows=[]
    for item in manifest['items']:
        note=notes[item['id']];assert sha(ROOT/note['panel'])==note['panel_sha256']==item['panel']['sha256']
        assert note['viewed_all3_candidates'] and note['native_review_pending'] and not note['label_admitted']
        letter=note['candidate_letter_for_native_followup'];candidate_id=item['anonymous_candidate_order'][ord(letter)-65]
        candidate=next(c for c in item['candidates'] if c['id']==candidate_id)
        assert sha(ROOT/item['image'])==item['input_sha256'] and sha(ROOT/candidate['path'])==candidate['sha256']
        before=Image.open(ROOT/item['image']).convert('RGB');after=Image.open(ROOT/candidate['path']).convert('RGB')
        assert before.size==after.size and max(before.size)<=1024
        w,h=before.size;canvas=Image.new('RGB',(2*w,h+72),'white');draw=ImageDraw.Draw(canvas)
        draw.text((4,4),item['id']+' | Native pixels, no admitted label',font=font(18),fill='black')
        draw.text((4,36),'INPUT',font=font(19),fill='black');draw.text((w+4,36),'CANDIDATE '+letter+' | pending',font=font(19),fill='black')
        canvas.paste(before,(0,72));canvas.paste(after,(w,72))
        path=target.parent/(item['id']+'.png');path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path,compress_level=1)
        rows.append(dict(id=item['id'],proposed_action=note['proposed_action'],candidate_id=candidate_id,candidate_letter=letter,
            input=dict(path=item['image'],sha256=item['input_sha256'],size=[w,h]),candidate=candidate,
            panel=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),size=list(canvas.size)),
            resampled_for_pair=False,cropped_for_pair=False,labels_admitted=False,review_status='not_viewed'))
    write(target,dict(at=stamp(),candidate_manifest_sha256=sha(source),triage_notes_sha256=sha(notes_path),rows=rows,
        scope='Full current1024max training-condition pixels, not full pre-resize original photograph. Candidate source PNGs copied without scaling/cropping into lossless pairs. All3 attempts were seen on whole-frame contact sheets; selected native pair is either proposed edit or restrained failed witness. No impossibility proof, no automatic label admission.',GPU_hours=0))
    print('64 full-input native-pixel pairs prepared;0new grades or admissions.')


if __name__=='__main__':main()

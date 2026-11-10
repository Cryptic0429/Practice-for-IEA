"""Original-size risk followups kept separately from candidate-panel observations."""
from pathlib import Path
import json
import hashlib
import datetime as dt

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'experiments/functional-v11-20261106'


def record(cid,paths,evidence,judgments=None):
    path=EXP/'candidate-full-resolution-observations.jsonl'
    old=path.read_text() if path.exists() else ''
    assert cid not in {r['id'] for r in map(json.loads,old.splitlines())}
    images=[]
    for relative in paths:
        file=EXP/relative
        images.append(dict(path=str(file.relative_to(ROOT)),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    row=dict(id=cid,at=dt.datetime.now(dt.timezone.utc).isoformat(),images=images,evidence=evidence,
        reviewer='Codex direct original-size visual AI review; not human',
        supplements_panels=True,training_dataset_admission=False,judgments=judgments or {},
        scope='Viewed actual PNGs, no resizing or adjustment. Current-tool reliability judgment, not an exhaustive parameter-space proof or real-photo safety certification.')
    path.write_text(old+json.dumps(row,ensure_ascii=False)+'\n')

if __name__=='__main__':
    record('visual-risk-01',[
        'counterfactual-candidates/images/visual-risk-01-noisy-input.png',
        'extended-candidates/images/visual-risk-01-severe-light-damped.png',
        'extended-candidates/images/visual-risk-01-mild-noise-light.png',
        'style-candidates/images/visual-risk-01-normal-dark-style.png',
        'counterfactual-candidates/images/visual-risk-01-canonical.png',
        'style-candidates/images/visual-risk-01-normal-compound.png',
        'extended-candidates/images/visual-risk-01-deep-compound.png',
        'extended-candidates/images/visual-risk-01-severe-compound-gentle.png'],
        'At actual417x627 size, the strong synthetic input already has coarse chroma blocks over forehead, face, vest, jeans and vegetation. The damped lift makes these blocks and a grey veil clearly visible; the gentle compound attempt also retains the destructive grid, so none of these tried strong-corruption outputs is basically acceptable. This distinguishes original corruption from added visibility and does not claim the model created the original blocks. Mild-grain brightening keeps natural face, plaid, vest and rail structure without conspicuous noise expansion. Normal and deep compound outputs visibly improve contrast/colour and gently lift the subject without severe facial shadow loss. The strongest style-darkened normal image is clearly darker while facial outline, hands, plaid pattern, denim and rails remain readable. These full-size checks support the provisional candidate judgments, pending final dataset/input/replay audits.')

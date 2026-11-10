"""Record inspected candidate stages without silently admitting data."""
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'experiments/functional-v10-20261106'


def record(cid,stage,judgments,evidence):
    path=EXP/'candidate-stage-observations.jsonl'
    prior=path.read_text() if path.exists() else ''
    assert (cid,stage) not in {(r['id'],r['stage']) for r in map(json.loads,prior.splitlines())}
    if stage in ('light-alternatives','compound'):
        manifest=EXP/'extended-candidates/manifest.json'
        c=next(c for c in json.loads(manifest.read_text())['cases'] if c['id']==cid)
        sheet=c['panels'][stage]
    else:
        assert stage=='normal-and-dark-style'
        manifest=EXP/'style-candidates/manifest.json'
        c=next(c for c in json.loads(manifest.read_text())['cases'] if c['id']==cid)
        sheet=c['panel']
    assert hashlib.sha256((ROOT/sheet['path']).read_bytes()).hexdigest()==sheet['sha256']
    row=dict(id=cid,stage=stage,at=dt.datetime.now(dt.timezone.utc).isoformat(),judgments=judgments,evidence=evidence,
        sheet=sheet,manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        reviewer='Codex direct visual AI review; not human',training_admission=False,
        note='These are actual viewed candidate assessments. Final data admission, reference-free input audit, target replay, original-size risk followups and frozen protocol still required. No L1 selection.')
    path.write_text(prior+json.dumps(row,ensure_ascii=False)+'\n')

if __name__=='__main__':
    for cid in ('visual-risk-01','visual-risk-02'):
        record(cid,'light-alternatives',{'mild-noise-light':['completed','acceptable'],
            'severe-light-damped':['completed','major_issue'],'severe-light-shadow-damped':['completed','major_issue']},
            'Mild grain remains unobtrusive after the substantial lift and subject colour/detail remain plausible. Both compensated strong-corruption attempts expose persistent large colour blocks or bands and add a washed grey appearance; neither supplies a basically acceptable natural result. This is AI judgment on tried settings, not an exhaustive impossibility proof.')
        record(cid,'compound',{'deep-compound':['completed','acceptable'],'shallow-compound':['completed','acceptable'],
            'mild-noise-compound':['completed','acceptable'],'severe-compound':['completed','major_issue'],
            'severe-compound-gentle':['partial','major_issue'],'severe-compound-shadows':['completed','major_issue']},
            'Clean and mildly noisy inputs gain stronger tonal separation and vivid colour with a modest lift, retaining recognizable face, clothing and foliage layers. All three strong-corruption attempts retain conspicuous non-natural coloured blocks/bands across subject and background; the gentle setting also gives only weak requested change. Original-size risk confirmation remains pending.')
        record(cid,'normal-and-dark-style',{'normal-light':['completed','acceptable'],'normal-compound':['completed','acceptable'],
            'normal-dark-style':['completed','acceptable'],'bright-dark-style':['completed','acceptable']},
            'The normal input is visibly brightened under the light request; the compound result has stronger colour/contrast and a modest lift without obvious severe loss. Both style-darkening outputs are substantially darker relative to their own inputs while the face, clothing pattern and surrounding structure remain recognizable with plausible colour. Opposite requests intentionally produce different appearances rather than always returning to canonical exposure.')

"""Lock the sole full-development qualifier, then repeat the same frozen training."""
import argparse,shutil,subprocess,sys
from .common import ROOT,EXP,protocol,read,write,sha,stamp
from .visual_review import EVAL


def lock():
    p=protocol();gate=read(EXP/'stage1-gate.json');eligible=[];evidence={}
    for model in gate['eligible']:
        path=EVAL/'development'/model/'direct-functional-summary.json';s=read(path)
        assert s['status']=='completed'
        passed=s['fully_completed_both_languages']>=81 and s['major_quality_issues']==0
        evidence[model]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),complete=s['fully_completed_both_languages'],major=s['major_quality_issues'],passed=passed)
        if passed:eligible.append(model)
    assert eligible==['magnitude-quality-e20'],'Multiple qualifiers require the declared blinded selection; no shortcut'
    path=EXP/'replication-lock.json';assert not path.exists()
    write(path,dict(at=stamp(),model=eligible[0],policy='magnitude-quality',epoch=20,
        initial_seed=20261205,replicate_seed=p['train']['replicate_seed'] if 'train' in p else p['training']['replicate_seed'],
        initial_head='experiments/functional-v21-20261107/runs/seed-20261205/checkpoint-20.pt',
        initial_head_sha256=sha(EXP/'runs/seed-20261205/checkpoint-20.pt'),
        protocol_sha256=sha(EXP/'training-protocol.json'),stage1_gate_sha256=sha(EXP/'stage1-gate.json'),
        controlled_evidence=evidence,selection_reason='Only one policy/checkpoint passed all declared development gates. No tied candidates require pairwise tie-breaking.',
        replication='Same80epoch frozen CPU training, evaluate only the locked20epoch checkpoint andquality policy; no switching to80 if replication fails.',
        acceptance_opened=False,L1_used=False))


def evaluate(splits):
    protocol();binding=read(EXP/'replication-lock.json');assert binding['model']=='magnitude-quality-e20' and binding['epoch']==20
    assert binding['protocol_sha256']==sha(EXP/'training-protocol.json')
    head=EXP/'runs/seed-20261206/checkpoint-20.pt';assert read(head.parent/'status.json')['status']=='completed'
    rep=EXP/'replication';rep.mkdir(exist_ok=True)
    for name in ('training-protocol.json','texture-features.json','stage1-gate.json','controlled-feature-binding.json'):
        dest=rep/name
        if dest.exists():assert sha(dest)==sha(EXP/name)
        else:shutil.copy2(EXP/name,dest)
    from . import evaluate as frozen
    frozen.EXP=rep
    sys.argv=['frozen-R21-evaluate','--model',binding['model'],'--policy',binding['policy'],
        '--head',str(head.relative_to(ROOT)),'--splits',*splits]
    frozen.main()


def prepare(splits):
    import json
    from functional_v6 import visual_review as established
    from .visual_review import configure,decisions
    rep=EXP/'replication';configure();established.EXP=rep;established.EVAL=rep/'evaluation'/EVAL.name
    for split in splits:
        if split=='decision-development':continue
        final_name={'old-native':'old-screen-final','native-development':'native-candidate-detail-final','development':'controlled-native-final'}[split]
        path=EXP/'review-derivation'/final_name/'review'/split/'codex-observations.jsonl'
        assert path.is_file(),'Inherit the final raw-plus-native derivation, not tentative sheet grades'
        dest=rep/'review'/split/path.name;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
        rows=[dict(json.loads(line),replication_exact_pixel_reuse_source=dict(path=str(path.relative_to(ROOT)),sha256=sha(path)),
            reuse_scope='Only exact input/request/criteria/output keys may match. No new independent review claimed.') for line in path.read_text().splitlines()]
        dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        (dest.parent/'codex-full-resolution-observations.jsonl').write_text('')
        established.prepare(split,['magnitude-quality-e20'],'locked-replicate')
    if 'decision-development' in splits:
        from . import visual_review as visual
        original=visual.EVAL;visual.EVAL=established.EVAL
        try:visual.decisions(['magnitude-quality-e20'])
        finally:visual.EVAL=original


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['lock','train','evaluate','prepare'])
    ap.add_argument('--splits',nargs='+',default=['old-native','native-development','decision-development','development']);args=ap.parse_args()
    if args.command=='lock':lock()
    elif args.command=='train':
        assert (EXP/'replication-lock.json').exists()
        subprocess.run([sys.executable,'-u','-m','functional_v21.train','--seed','20261206'],cwd=ROOT,check=True)
    elif args.command=='evaluate':evaluate(args.splits)
    else:prepare(args.splits)

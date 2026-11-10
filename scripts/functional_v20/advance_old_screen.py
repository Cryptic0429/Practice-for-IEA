"""Only actual old-screen qualifiers advance; no refitting or GPU allocation."""
import subprocess,sys
from .common import ROOT,EXP,read,write,sha,stamp
from .visual_review import EVAL,configure,decisions
from functional_v6 import visual_review as established

def main():
    rows={};eligible=[]
    for route in ('last','request_mean'):
        for epoch in (20,80):
            mid=f'{route.replace("_","-")}-e{epoch}';path=EVAL/'old-native'/mid/'direct-functional-summary.json';s=read(path)
            assert s['status']=='completed'
            passed=s['fully_completed_both_languages']>=4 and s['major_quality_issues']==0
            rows[mid]=dict(completed=s['fully_completed_both_languages'],major=s['major_quality_issues'],eligible=passed,summary_sha256=sha(path))
            if passed:eligible.append((mid,route,epoch))
    write(EXP/'old-screen-gate.json',dict(at=stamp(),status='completed',results=rows,eligible=[r[0] for r in eligible],
          frozen_protocol_sha256=sha(EXP/'training-protocol.json'),used_L1=False,AI_review_only=True))
    for mid,route,epoch in eligible:
        subprocess.run([sys.executable,'-u','-m','functional_v20.evaluate','--model',mid,'--checkpoint',
           f'experiments/functional-v20-20261107/runs/{route}/checkpoint-{epoch}.pt',
           '--splits','native-development','decision-development'],cwd=ROOT,check=True)
    if eligible:
        configure();models=[r[0] for r in eligible]
        established.prepare('native-development',models,'old-qualified-checkpoints')
        decisions(models)
    print('Advanced', [r[0] for r in eligible],'; no new training or GPU.')

if __name__=='__main__':main()

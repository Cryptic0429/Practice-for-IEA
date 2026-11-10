"""Require actual old-request AI gate before further fixed development inference."""
import subprocess,sys
from .common import ROOT,EXP,POLICIES,read,write,sha,stamp
from .visual_review import EVAL,configure,decisions
from functional_v6 import visual_review as established

def main():
    scores={};eligible=[]
    for epoch in (20,80):
        for policy in POLICIES:
            mid=f'{policy}-e{epoch}';path=EVAL/'old-native'/mid/'direct-functional-summary.json';s=read(path)
            assert s['status']=='completed'
            passed=s['fully_completed_both_languages']>=4 and s['major_quality_issues']==0
            scores[mid]=dict(completed=s['fully_completed_both_languages'],major=s['major_quality_issues'],eligible=passed,summary_sha256=sha(path))
            if passed:eligible.append((mid,policy,epoch))
    write(EXP/'old-screen-gate.json',dict(at=stamp(),status='completed',results=scores,eligible=[r[0] for r in eligible],
        training_protocol_sha256=sha(EXP/'training-protocol.json'),L1_used=False,AI_review_only=True))
    for mid,policy,epoch in eligible:
        subprocess.run([sys.executable,'-u','-m','functional_v21.evaluate','--model',mid,'--policy',policy,
            '--head',f'experiments/functional-v21-20261107/runs/seed-20261205/checkpoint-{epoch}.pt',
            '--splits','native-development','decision-development'],cwd=ROOT,check=True)
    if eligible:
        configure();models=[r[0] for r in eligible]
        established.prepare('native-development',models,'old-qualified-policies')
        decisions(models)
    print('Advanced', [r[0] for r in eligible],'; noGPU andno newtraining.')

if __name__=='__main__':main()

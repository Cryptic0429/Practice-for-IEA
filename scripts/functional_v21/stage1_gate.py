"""Bind functional development evidence before any controlled feature extraction."""
from .common import EXP,read,write,sha,stamp,protocol
from .visual_review import EVAL


def main():
    protocol();assert not (EXP/'stage1-gate.json').exists()
    eligible=[];results={}
    for epoch in (20,80):
        for policy in ('magnitude','magnitude-quality'):
            model=f'{policy}-e{epoch}';rows={}
            for split in ('old-native','native-development','decision-development'):
                path=EVAL/split/model/('decision-summary.json' if split=='decision-development' else 'direct-functional-summary.json')
                s=read(path);assert s['status']=='completed'
                rows[split]=dict(path=str(path.relative_to(EXP.parents[1])),sha256=sha(path),
                    completed=s.get('fully_completed_both_languages',s.get('passed_both_languages')),major=s.get('major_quality_issues',0))
            passed=(rows['old-native']['completed']>=4 and rows['native-development']['completed']>=16
                and rows['decision-development']['completed']==20 and not any(r['major'] for r in rows.values()))
            results[model]=dict(eligible=passed,evidence=rows)
            if passed:eligible.append(model)
    write(EXP/'stage1-gate.json',dict(at=stamp(),training_protocol_sha256=sha(EXP/'training-protocol.json'),
        results=results,eligible=eligible,L1_used=False,AI_review_only=True,
        scope='Native followups completed for the sole potentially qualifying policy. Other major-failure variants retain unresolved quality uncertainty and conservative completion lower bounds.'))
    print('Controlled stage eligible:',eligible)


if __name__=='__main__':main()

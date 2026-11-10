"""Reuse final reviews only for exact input/request/criteria/output identities."""
import copy
import json
import re
import shutil
from functional_v1.common import ROOT, read, write, sha, stamp
from .common import EXP


def main():
    assert not (EXP/'review-reuse.json').exists()
    receipt=[]
    for split in ('native-development','old-native','development'):
        source_paths=[]
        for path in (ROOT/'experiments').glob('functional-v*-*/evaluation/*/*/*/direct-review.json'):
            match=re.match(r'functional-v(\d+)-',path.relative_to(ROOT/'experiments').parts[0])
            if match and 1<=int(match.group(1))<=20 and path.parent.parent.name==split:
                source_paths.append(path)
        groups={}
        for path in sorted(source_paths):
            for item in read(path):
                note=item['assessment'];key=item.get('item_id')
                if not key or not note.get('image_sha256'):continue
                assert key==note['item_id']
                groups.setdefault(key,[]).append((path,note))
        notes=[]
        for key,prior in groups.items():
            assert len({n['image_sha256'] for _,n in prior})==1
            note=copy.deepcopy(prior[-1][1])
            assert sha(ROOT/note['sheet'])==note['sheet_sha256']
            disputed=[f for f in ('completion','quality') if len({n[f] for _,n in prior})>1]
            for field in disputed:note[field]='uncertain'
            if disputed:note['evidence']='Exact same input/request/criteria/output has conflicting final AI assessments on '+','.join(disputed)+'. Prior evidence is preserved; no human adjudication, disputed fields stay uncertain.'
            note['prior_final_sources']=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p,_ in prior]
            note['reuse_policy']='Exact semantic/input/output key only. No new viewing or independent user approval; all final same-render disagreements retained.'
            notes.append(note)
        path=EXP/'review'/split/'codex-observations.jsonl'
        # The editor control has already contributed final notes in these files.
        if path.exists():
            snapshot=EXP/'editor-control-review-snapshot'/split/path.name
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            if not snapshot.exists():shutil.copy2(path,snapshot)
            assert sha(snapshot)==read(EXP/'editor-control-summary.json')['summaries'][split]['review_sha256']
        existing={n['item_id']:n for n in map(json.loads,path.read_text().splitlines())} if path.exists() else {}
        for note in notes:existing[note['item_id']]=note
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(n,ensure_ascii=False)+'\n' for n in existing.values()))
        native=path.with_name('codex-full-resolution-observations.jsonl');assert not native.exists();native.write_text('')
        receipt.append(dict(split=split,notes=len(existing),initial_sha256=sha(path),sources={str(p.relative_to(ROOT)):sha(p) for p in source_paths}))
    write(EXP/'review-reuse.json',dict(at=stamp(),records=receipt,
        policy='Final AI notes only, exact key reuse; disputed grades remain uncertain. No automatic quality rating or new independent review.'))


if __name__=='__main__':main()

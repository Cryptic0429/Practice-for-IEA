"""Carry forward only exact, final AI image assessments with explicit provenance."""
import json
from functional_v1.common import ROOT, read, write, sha, stamp
from .runtime import EXP


def main():
    receipt=EXP/'review-reuse.json'; assert not receipt.exists()
    records=[]
    for split in ('native-development','old-native'):
        source=ROOT/'experiments/functional-v9-20261106/review-derivation/all-checkpoints-final/review'/split/'codex-observations.jsonl'
        notes=[json.loads(line) for line in source.read_text().splitlines()]
        assert len({n['item_id'] for n in notes})==len(notes)
        destination=EXP/'review'/split/'codex-observations.jsonl'
        assert not destination.exists(); destination.parent.mkdir(parents=True,exist_ok=True)
        for note in notes:
            assert sha(ROOT/note['sheet'])==note['sheet_sha256']
            note['prior_final_observation_source']=dict(path=str(source.relative_to(ROOT)),sha256=sha(source))
            note['reuse_policy']='Exact input/request/criteria/output item key only. Inherited AI observations, including uncertainties; not new viewing or human feedback.'
        destination.write_text(''.join(json.dumps(n,ensure_ascii=False)+'\n' for n in notes))
        (destination.parent/'codex-full-resolution-observations.jsonl').write_text('')
        records.append(dict(split=split,source=str(source.relative_to(ROOT)),source_sha256=sha(source),
                            destination=str(destination.relative_to(ROOT)),initial_sha256=sha(destination),observations=len(notes)))
    write(receipt,dict(at=stamp(),records=records,policy='Only exact key matches may be reused. Newly rendered unique outputs require direct anonymous viewing. Reused uncertainty cannot be silently upgraded. All sources retained.'))

if __name__=='__main__':
    main()

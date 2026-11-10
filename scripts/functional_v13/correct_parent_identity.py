"""Correct only a copied summarizer's inference flag, preserving the first artifact."""
import shutil
from functional_v1.common import read,write,sha,stamp
from functional_v11.outcomes import EDITOR_VERSION
from .runtime import EXP,PARENT_SHA


def main():
    receipt=EXP/'parent-summary-inference-metadata-correction.json';assert not receipt.exists()
    records=[]
    for split in ('native-development','old-native'):
        folder=EXP/'evaluation'/EDITOR_VERSION/split/'parent-frozen'
        identity=read(folder/'identity.json');assert identity['kind']=='exact_frozen_predictions_reused_no_new_inference'
        assert identity['adapter_sha256']==PARENT_SHA and identity['model_forward_calls']==0
        path=folder/'direct-functional-summary.json';before=folder/'summary-before-inference-metadata-correction.json'
        assert not before.exists();shutil.copy2(path,before)
        d=read(path);prior=read(before)
        d.update(inference_performed=False,runtime='Exact frozen R11 predictions and pixels reused; no new model inference',
            inference_metadata_correction=str(receipt.relative_to(EXP.parents[1])))
        write(path,d)
        assert all(d[k]==v for k,v in prior.items() if k not in ('inference_performed','runtime'))
        records.append(dict(split=split,before=str(before),before_sha256=sha(before),after=str(path),after_sha256=sha(path)))
    write(receipt,dict(at=stamp(),scope='Only native/old parent inference metadata; no visual judgment, output, completion count or controlled-parent actual inference changed.',
        cause='Copied frozen summarizer recognises only the older CPU replay kind; the new exact-prediction reuse kind was incorrectly flagged as inference.',records=records))


if __name__=='__main__':main()

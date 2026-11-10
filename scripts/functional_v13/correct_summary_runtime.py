"""Correct a copied R12 display field; preserve original summaries and all assessments."""
import copy
from functional_v1.common import ROOT,read,write,sha,stamp
from .runtime import EXP


def main():
    receipt=EXP/'summary-runtime-metadata-correction.json';assert not receipt.exists()
    changes=[]
    for path in sorted((EXP/'evaluation').glob('*/*/*/direct-functional-summary.json')):
        value=read(path)
        if value.get('runtime')!='functional_v12.runtime':continue
        before=copy.deepcopy(value);relative=path.relative_to(EXP)
        saved=EXP/'metadata-before-runtime-correction'/relative;write(saved,before)
        value['runtime']='functional_v13.runtime'
        assert value['inference_performed'] is True
        check=copy.deepcopy(value);check['runtime']=before['runtime'];assert check==before
        write(path,value)
        changes.append(dict(path=str(path.relative_to(ROOT)),before_sha256=sha(saved),after_sha256=sha(path)))
    assert len(changes)==11
    selection_path=EXP/'development-selection.json';selection=read(selection_path)
    write(EXP/'metadata-before-runtime-correction/development-selection.json',selection)
    for mid,meta in selection['models'].items():
        for key,split in [('native_development_summary','native-development'),('old_native_summary','old-native'),('controlled_development_summary','development')]:
            meta[key]=read(EXP/'evaluation'/meta['editor']/split/meta['evaluation_id']/'direct-functional-summary.json')
        target=ROOT/meta['adapter']/'metadata.json'
        write(EXP/'metadata-before-runtime-correction/models'/mid/'metadata.json',read(target));write(target,meta)
    selection['parent_controlled_summary']=read(EXP/'evaluation'/next(iter(selection['models'].values()))['editor']/'development/parent-frozen/direct-functional-summary.json')
    selection['summary_runtime_metadata_correction']=str(receipt.relative_to(ROOT));write(selection_path,selection)
    write(receipt,dict(at=stamp(),changes=changes,reason='Copied finalizer labelled actual R13 forward passes as functional_v12.runtime. Actual inference was R13, as its identity/runtime hashes and launch commands prove.',
        scope='Display runtime field only; judgments, completion/risk counts, raw notes, decisions, images, inference status and frozen code/protocols unchanged.',
        initial_selection_snapshot_preserved=True,registered_model_metadata_corrected=True))
    protocol=read(EXP/'training-protocol.json');assert all(sha(ROOT/p)==h for p,h in protocol['code_hashes'].items())
    print('11 runtime display fields corrected with immutable originals; functional results unchanged.')


if __name__=='__main__':main()

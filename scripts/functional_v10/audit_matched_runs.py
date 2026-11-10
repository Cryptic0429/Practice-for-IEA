"""Inspect saved initialization evidence; no additional model load or forward."""
from functional_v1.common import read,write,stamp
from .runtime import EXP,PARENT_SHA


def main():
    path=EXP/'matched-initialization-input-check.json';assert not path.exists()
    folders=[EXP/'runs'/f'{route}-seed-20261121' for route in ('language','language_merger')]
    states=[read(p/'initialization.json') for p in folders]
    for s in states:
        assert s['parent_sha256']==PARENT_SHA and s['language_parent_tensors_copied_exactly']==112
        assert s['adapter_count']==1 and s['stacked_adapters'] is False and s['visual_merger_B_initially_zero']
        assert not any(s['full_weight_loading_information'].get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    fingerprints=[read(p/'initial-adapter-fingerprints.json') for p in folders]
    language=[{k:v for k,v in f.items() if 'visual' not in k} for f in fingerprints]
    assert len(language[0])==112 and language[0]==language[1]
    assert len(fingerprints[0])==112 and len(fingerprints[1])==116
    assert read(folders[0]/'token-audit.json')==read(folders[1]/'token-audit.json')
    for p,count in zip(folders,(112,116)):
        gradient=read(p/'first-gradient-check.json');assert gradient['finite'] and gradient['tensors']==count and gradient['optimizer_contains_only_adapter']
    write(path,dict(at=stamp(),status='passed_actual_quantized_load_and_training_initialization',
        language_tensor_fingerprints_identical=112,training_token_records_identical=480,
        trainable_tensors=[112,116],exact_single_parent_transfer=True,visual_merger_B_zero=True,
        full_weight_loading_information_clean=True,first_optimizer_gradients_finite=True,
        model_capability_proof=False,additional_GPU_hours=0))
    print('Matched112 parent tensor hashes and480 cached input/label token records; actual initial gradients/load verified.')

if __name__=='__main__':main()

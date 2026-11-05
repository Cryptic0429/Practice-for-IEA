"""Verify measured-pixel semantics, same visual resize, and metadata leakage boundaries."""
import copy
import numpy as np
import torch
from PIL import Image
from functional_v1.common import image_path
from functional_v1.runtime import image_input
from .common import *
from .runtime import measurements,processor,encode,build_batch


def main():
    black=measurements(Image.new('RGB',(64,64),'black'));white=measurements(Image.new('RGB',(64,64),'white'))
    assert black['luminance_mean']==0 and white['luminance_mean']==1000
    assert black['luminance_std']==white['luminance_std']==0 and black['saturation_mean']==white['saturation_mean']==0
    assert black['black_clip_fraction_L_below_002']==white['white_clip_fraction_L_above_098']==1000
    cases=read(DATA/'cases.json');case=cases[0];p=processor();poison=copy.deepcopy(case)
    poison.update(parameters={'exposure':-100},target='DO_NOT_LEAK_TARGET',canonical_photo='DO_NOT_LEAK_CANONICAL',review='DO_NOT_LEAK_REVIEW',statistics={'luminance_mean':999},action='DO_NOT_LEAK_ACTION',expected_change='DO_NOT_LEAK_EXPECTED',group_id='DO_NOT_LEAK_GROUP')
    for route in ('plain','measured'):
        a=encode(p,case,'zh',route);b=encode(p,poison,'zh',route)
        assert set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)
        decoded=p.decode(a['input_ids'][0]);assert 'DO_NOT_LEAK' not in decoded
        for language in ('zh','en'):
            batch=build_batch(p,case,language,case['parameters'],route);assert int(batch['labels'].ne(-100).sum())>4
    a=measurements(image_input(image_path(case)))
    darker=next(c for c in cases if c['source_id']==case['source_id'] and c['action']=='restore-exposure-strong')
    b=measurements(image_input(image_path(darker)))
    assert case['instructions']==darker['instructions'] and a['luminance_mean']>b['luminance_mean']
    write(EXP/'input-audit.json',{'status':'passed','at':stamp(),'poison_metadata_changes_input':False,'routes':['plain','measured'],'bilingual_prefix_mask_checked':True,'same_request_different_image_observations':{'mild':a,'strong':b},'claims':'Features are input observations, never acceptance scores. Full final results require direct visual review.'})
    print('Measured pixel semantics, bilingual assistant mask, metadata poison isolation and same-request image conditioning passed')
if __name__=='__main__':main()

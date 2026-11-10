"""Reuse exactly matching CPU batches; rebuild only the 22 repaired JSON rows."""
import os
import torch
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v13 import runtime as old_runtime
from .runtime import EXP, DATA, processor, build_batch
from .outcomes import input_row, system_prompt, target_text
from functional_v11.outcomes import input_row as old_input_row, system_prompt as old_prompt


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (EXP/'batches.pt').exists()
    prior=ROOT/'data/training/functional-v13-20261106'
    cache=read(ROOT/'experiments/functional-v13-20261106/batch-cache.json')
    assert sha(ROOT/cache['cache_path'])==cache['cache_sha256']
    assert sha(prior/'train-accepted.json')==cache['dataset_sha256']
    assert sha(prior/'cases.json')==cache['cases_sha256']
    assert processor is old_runtime.processor and build_batch is old_runtime.build_batch
    assert system_prompt()==old_prompt()
    old_rows=read(prior/'train-accepted.json'); rows=read(DATA/'train-accepted.json')
    old_cases={c['id']:c for c in read(prior/'cases.json')}
    cases={c['id']:c for c in read(DATA/'cases.json')}
    batches=torch.load(ROOT/cache['cache_path'],map_location='cpu',weights_only=True)
    assert len(batches)==len(rows)==len(old_rows)==1850
    p=processor(); prepared=[]; tokens=[]; changed=[]; inherited=[]
    for old,row,batch in zip(old_rows,rows,batches):
        assert old['id']==row['id'] and old['case_id']==row['case_id'] and old['language']==row['language']
        c=cases[row['case_id']]; prior_c=old_cases[row['case_id']]; lang=row['language']
        assert sha(ROOT/c['image'])==c['input_sha256']==prior_c['input_sha256']
        assert input_row(c,lang)==old_input_row(prior_c,lang)
        if row['decision']==old['decision']:
            result=batch; inherited.append(row['id'])
        else:
            result=build_batch(p,c,lang,row['decision']); changed.append(row['id'])
            old_mask=batch['labels'].eq(-100); new_mask=result['labels'].eq(-100)
            assert torch.equal(batch['input_ids'][old_mask],result['input_ids'][new_mask])
            assert torch.equal(batch['pixel_values'],result['pixel_values'])
            assert torch.equal(batch['image_grid_thw'],result['image_grid_thw'])
        assert all(t.device.type=='cpu' for t in result.values() if isinstance(t,torch.Tensor))
        prepared.append(dict(result))
        tokens.append(dict(id=row['id'],action=row['decision']['action'],
            input_tokens=int(result['input_ids'].numel()),assistant_tokens=int(result['labels'].ne(-100).sum())))
    assert len(changed)==22 and len(inherited)==1828
    torch.save(prepared,EXP/'batches.pt')
    write(EXP/'batch-cache.json',dict(at=stamp(),status='CPU_only_exact_input_cache_inherited_and_repaired_labels_rebuilt',
        dataset_sha256=sha(DATA/'train-accepted.json'),cases_sha256=sha(DATA/'cases.json'),
        cache_path=str((EXP/'batches.pt').relative_to(ROOT)),cache_sha256=sha(EXP/'batches.pt'),rows=len(prepared),tokens=tokens,
        inherited_rows=inherited,rebuilt_rows=changed,source_cache=cache,
        repaired_rows_prefix_and_pixels_exactly_equal=True,processor_and_builder_identical_function_objects=True,
        model_forward_calls=0,GPU_hours=0,limitation='Input/data integrity only, not model prediction quality.'))
    print('CPU cache prepared:1828 exact inherited rows,22 repaired rows,all1850 input whitelist checks')


if __name__=='__main__':main()

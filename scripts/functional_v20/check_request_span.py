"""CPU feasibility only: literal-span tokenization, no feature cache or training."""
from functional_v1.common import ROOT,read,write,stamp,sha

def main():
    import torch
    from functional_v15.runtime import processor
    from functional_v15.outcomes import input_row
    from functional_v1.runtime import encode
    from .request_span import rendered_request,request_indices
    torch.set_num_threads(2)
    source=ROOT/'data/training/functional-v19-20261107/cases.json'
    cases=read(source);p=processor();count=0
    for c in cases:
        for language in ('zh','en'):
            row=input_row(c,language)
            _,indices,_,_=rendered_request(p,row,c['instructions'][language])
            assert indices;count+=1
    chosen=['train-020-restore-exposure-strong','train-100-restore-contrast-strong','train-064-restore-saturation-strong']
    by_id={c['id']:c for c in cases};checks=[]
    for cid in chosen:
        c=by_id[cid]
        for language in ('zh','en'):
            row=input_row(c,language);actual=encode(p,row)
            indices=request_indices(p,row,c['instructions'][language],actual)
            checks.append(dict(case_id=cid,language=language,literal_tokens=len(indices),prefix_tokens=actual['input_ids'].shape[1],
                actual_prefix_expansion_exact=True))
    assert not torch.cuda.is_initialized()
    out=ROOT/'experiments/functional-v20-20261107/request-span-feasibility.json';out.parent.mkdir(parents=True,exist_ok=True)
    write(out,dict(at=stamp(),status='CPU_literal_request_span_feasible_no_training',source_cases_sha256=sha(source),
        template_language_inputs_checked=count,actual_encoded_prefixes_checked=len(checks),actual_checks=checks,
        source_code_sha256=sha(ROOT/'scripts/functional_v20/request_span.py'),GPU_calls=0,model_loaded=False,
        feature_cache_created=False,controller_trained=False,development_or_acceptance_images_read=False,
        limitation='Tokenizer/prefix feasibility only. No claim of better representations, model functionality, image quality or route selection. R19 remains unchanged.'))
    print('CPU literal spans',count,'; actual target-free image-expanded prefixes',len(checks),'; no7B model, cache, training or GPU.')

if __name__=='__main__':main()

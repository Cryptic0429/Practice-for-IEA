"""Exact frozen v1 runtime; candidate outputs cannot tune heldout criteria or labels."""
import argparse,json,os
from pathlib import Path
import numpy as np
from functional_v1.common import ROOT,read,write,append,sha,stamp,GPUJob,EDITOR_VERSION
from functional_v1.editor import load_rgb,edit,save,statistics
from .source_bank import EXP,DATA

EVAL=EXP/'evaluation'/EDITOR_VERSION
KEYS={'native-development':'native_development','old-native':'old_native_regressions',
      'development':'controlled_development','native-acceptance':'native_acceptance','acceptance':'controlled_acceptance'}

def case_file(split):
    p=read(EXP/'evaluation-protocol.json')['case_files'][KEYS[split]]
    path=ROOT/p['path'];assert sha(path)==p['sha256'];return path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--adapter',type=Path)
    ap.add_argument('--splits',nargs='+',choices=list(KEYS),default=['native-development','old-native']);ap.add_argument('--gpu',choices=['0','2','3'],required=True)
    a=ap.parse_args();protocol=read(EXP/'training-protocol.json')
    runlock=read(EXP/'evaluation-run-lock.json')
    for f,digest in runlock['code_hashes'].items():assert sha(ROOT/f)==digest
    for f,digest in protocol['code_hashes'].items():assert sha(ROOT/f)==digest
    assert sha(EXP/'evaluation-protocol.json')==protocol['evaluation_protocol_sha256']
    adapter=a.adapter
    if adapter is not None:
        if not adapter.is_absolute():adapter=ROOT/adapter
        assert (adapter/'adapter_model.safetensors').exists()
    checksum=sha(adapter/'adapter_model.safetensors') if adapter else None
    identity_base=dict(model_id=a.model,adapter=str(adapter.relative_to(ROOT)) if adapter else None,adapter_sha256=checksum,
        editor=EDITOR_VERSION,runtime='functional_v1.runtime',evaluator_sha256=sha(__file__),
        core_code_hashes={f:sha(ROOT/f) for f in ('scripts/functional_v1/common.py','scripts/functional_v1/runtime.py','scripts/functional_v1/editor.py')},
        inputs='Only original input image, original single-language request, exact v1 tools. No note, statistics, target, reference, review or IDs.')
    queues=[]
    for split in a.splits:
        path=case_file(split);cases=read(path);out=EVAL/split/a.model;out.mkdir(parents=True,exist_ok=True)
        if 'acceptance' in split:
            lock=read(EXP/'final-model-lock.json');assert lock['models'][a.model]['adapter_sha256']==checksum
            assert lock['evaluation_protocol_sha256']==sha(EXP/'evaluation-protocol.json')
            for f,digest in lock['code_hashes'].items():assert sha(ROOT/f)==digest
        ident=dict(identity_base,cases_sha256=sha(path),split=split)
        if (out/'identity.json').exists():assert read(out/'identity.json')==ident
        else:write(out/'identity.json',ident)
        rows=list(map(json.loads,(out/'predictions.jsonl').read_text().splitlines())) if (out/'predictions.jsonl').exists() else []
        done={(r['case_id'],r['language']) for r in rows};assert len(done)==len(rows)
        if len(done)<len(cases)*2:queues.append((split,cases,out,rows,done))
    if not queues:return
    with GPUJob(a.gpu,'v6-evaluate-'+a.model,max_seconds=5400) as budget:
        import torch
        from functional_v1.runtime import load_model,processor,predict
        torch.set_num_threads(4);model=load_model(adapter);p=processor()
        for split,cases,out,rows,done in queues:
            for c in cases:
                assert sha(ROOT/c['image'])==c['input_sha256']
                for language in ('zh','en'):
                    if (c['id'],language) in done:continue
                    budget.tick();row=dict(case_id=c['id'],group_id=c['group_id'],family=c['family'],action=c['action'],
                        language=language,model_id=a.model,instruction=c['instructions'][language],input_sha256=c['input_sha256'],
                        input_image=c['image'],status='failed')
                    try:
                        response,params=predict(model,p,c,language);before=load_rgb(ROOT/c['image']);after=edit(before,params)
                        path=out/'images'/f'{c["id"]}-{language}.png';save(path,after)
                        row.update(status='executed',response=response,parameters=params,output_image=str(path.relative_to(ROOT)),
                            output_sha256=sha(path),input_output_identical=bool(np.array_equal(before,after)),
                            input_statistics=statistics(before),output_statistics=statistics(after))
                    except Exception as e:row['error']=f'{type(e).__name__}: {e}'
                    row['at']=stamp();append(out/'predictions.jsonl',row);rows.append(row);done.add((c['id'],language))
                    write(out/'status.json',dict(status='running',pid=os.getpid(),completed=len(done),total=len(cases)*2,at=stamp()))
            write(out/'status.json',dict(status='completed_predictions',completed=len(rows),total=len(cases)*2,
                executed=sum(r['status']=='executed' for r in rows),finished_at=stamp()))

if __name__=='__main__':main()

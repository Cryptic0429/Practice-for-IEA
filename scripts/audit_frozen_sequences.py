"""Verify frozen records fit the real Qwen template at the candidate resolution."""
import collections
import concurrent.futures
import copy
import json
from pathlib import Path

import numpy as np
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor

ROOT=Path('/mnt/sdc/zhc/IEA')
SPLIT=ROOT/'data/stage1/splits/image-edit-3k-v1'


def main():
    args=dict(model_name_or_path=str(ROOT/'models/Qwen2.5-VL-7B-Instruct'),trust_remote_code=False,
              image_max_pixels=262144,stage='sft',do_train=True,finetuning_type='lora',
              dataset='iea_edit_3k_train',dataset_dir=str(ROOT/'data'),template='qwen2_vl',cutoff_len=2048,
              output_dir=str(ROOT/'outputs/sequence-audit'),report_to='none',use_cpu=True)
    ma,da,ta,fa,ga=get_train_args(args)
    tm=load_tokenizer(ma)
    template=get_template_and_fix_tokenizer(tm['tokenizer'],da)
    all_rows={s:json.loads((SPLIT/f'{s}.json').read_text()) for s in ['train','validation','test']}
    paths=sorted({row['images'][0] for rows in all_rows.values() for row in rows})
    original=template.mm_plugin._get_mm_inputs
    def image_grid(path):
        result=original([path],[],[],tm['processor'])
        return path, {'image_grid_thw':result['image_grid_thw']}
    # Cache only the tiny image grids, never pixel tensors; image transforms use upstream code.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        grids=dict(pool.map(image_grid,paths))
    def cached(images,videos,audios,processor):
        assert len(images)==1 and not videos and not audios
        return grids[images[0]]
    template.mm_plugin._get_mm_inputs=cached
    long_da=copy.deepcopy(da)
    long_da.cutoff_len=32768
    full=SupervisedDatasetProcessor(template=template,data_args=long_da,**tm)
    short=SupervisedDatasetProcessor(template=template,data_args=da,**tm)
    records={}
    for split,rows in all_rows.items():
        records[split]=[]
        for row in rows:
            c=row['conversations']
            inputs=([c[1]],[c[2]],c[0]['content'],None,row['images'],[],[])
            ids,labels=full._encode_data_example(*inputs)
            actual_ids,actual_labels=short._encode_data_example(*inputs)
            assert ids==actual_ids and labels==actual_labels, f'Truncated record: {row["id"]}'
            target=sum(t!=-100 for t in labels)
            assert 0<target<len(ids)
            grid=grids[row['images'][0]]['image_grid_thw'][0]
            records[split].append(dict(id=row['id'],group_id=row['group_id'],tokens=len(ids),
                                       supervised_tokens=target,image_grid_thw=grid.tolist(),
                                       image_tokens=int(grid.prod())//4,truncated=False))
        print('AUDITED',split,len(rows),'max_tokens',max(r['tokens'] for r in records[split]),flush=True)
    summary={s:dict(rows=len(rs),truncated_rows=0,tokens_max=max(r['tokens'] for r in rs),
                    tokens_percentiles=dict(zip(['p50','p95','p99'],np.percentile([r['tokens'] for r in rs],[50,95,99]).tolist())),
                    min_supervised_tokens=min(r['supervised_tokens'] for r in rs)) for s,rs in records.items()}
    worst=sorted(records['train'],key=lambda r:(-r['tokens'],-r['image_tokens'],r['id']))[:16]
    lookup={r['id']:r for r in all_rows['train']}
    stress=[lookup[r['id']] for r in worst]
    path=SPLIT/'stress_train_262k.json'
    path.write_text(json.dumps(stress,indent=2,ensure_ascii=False)+'\n')
    regpath=ROOT/'data/dataset_info.json'
    registry=json.loads(regpath.read_text())
    entry=copy.deepcopy(registry['gier_sft_distill'])
    entry['file_name']=str(path)
    registry['iea_edit_3k_stress_train']=entry
    regpath.write_text(json.dumps(registry,indent=2)+'\n')
    report=dict(image_max_pixels=262144,cutoff_len=2048,summary=summary,stress_training_ids=[r['id'] for r in worst],
                no_model_outputs_computed=True,note='Formatting audit only; validation/test records never enter training or throughput probes.')
    (ROOT/'logs/frozen-sequence-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (SPLIT/'sequence-lengths-262k.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

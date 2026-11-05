import json
from pathlib import Path
from zip_ranges import extract_selected
from download_stage1_archives import FILES

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT/'metadata/iea-archive-index.json').read_text())
selected = [r for r in rows if
            r['name'].startswith('datasets/GIER-Summary/merge/') or
            r['name'] in {f'datasets/{task}/sft_{split}.json'
                          for task in ['GIER-Edit','GIER-Summary']
                          for split in ['train','test','train_info']} or
            (r['name'].startswith('data/GIER_distill/') and r['name'].endswith(('.json','.txt')))]
extract_selected(rows,selected,[dict(id=i,size=s) for i,_,s in FILES],
                 ROOT/'.downloads/stage1-release',ROOT/'.downloads/stage1-ranges')

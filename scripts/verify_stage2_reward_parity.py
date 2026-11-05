"""Compare real editor/reward results across environments on training-only probes."""
import argparse
import json
from pathlib import Path
from stage2_reward import ImageReward

ROOT=Path('/mnt/sdc/zhc/IEA')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--export-stage1',action='store_true');args=parser.parse_args()
    rows=json.loads((ROOT/'data/stage2/stage2-A-edit-grpo-v1/preflight.json').read_text())
    protocol=json.loads((ROOT/'configs/evaluation-protocol-v1.json').read_text())
    reward=ImageReward(ROOT,protocol,rows)
    results=[]
    for r in rows:
        for kind,params in [('target',r['extra_info']['label']),('zero',{})]:
            record=reward.score(r['id'],'```json\n'+json.dumps(params)+'\n```')
            results.append(dict(id=r['id'],kind=kind,reward=record['reward'],scoring=record['scoring']))
    out=ROOT/'logs/stage2-A-edit-grpo-v1'
    if args.export_stage1:
        target=out/'stage1-editor-probes.json';assert not target.exists()
        target.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    else:
        expected=json.loads((out/'stage1-editor-probes.json').read_text())
        result=dict(all_equal=expected==results,training_rows=len(rows),cases=len(results),model_forwards=0,
                    scope='Target parameters and empty parameters executed with identical per-ID random seeds')
        (out/'reward-parity.json').write_text(json.dumps(result,indent=2)+'\n')
        print(result);assert result['all_equal']


if __name__=='__main__':
    main()

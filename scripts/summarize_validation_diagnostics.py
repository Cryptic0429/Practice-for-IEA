"""Descriptive generation diagnostics on the fixed first 32 validation rows."""
from datetime import datetime, timezone
import json
from pathlib import Path
from summarize_evaluation import summarize_model

ROOT = Path('/mnt/sdc/zhc/IEA')


def main():
    protocol = json.loads((ROOT / 'configs/evaluation-protocol-v1.json').read_text())
    rows = json.loads((Path(protocol['split_directory']) / 'validation.json').read_text())[:32]
    summaries, pending = {}, []
    for model in protocol['models']:
        folder = ROOT / f'outputs/evaluation-v1/validation-{model}-diagnostics-v1'
        if not (folder / 'status.json').exists():
            pending.append(model)
            continue
        status = json.loads((folder / 'status.json').read_text())
        if status['status'] != 'completed':
            pending.append(model)
            continue
        predictions = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
        assert len(predictions) == 32 and [r['id'] for r in predictions] == [r['id'] for r in rows]
        summary = summarize_model(predictions, protocol['metrics']['comparison_tolerance'])
        (folder / 'diagnostic-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        summaries[model] = summary
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), split='validation',
                  selection='First 32 rows of the frozen validation list; identical IDs for every condition.',
                  sample_ids=[r['id'] for r in rows], models=summaries, pending_models=pending,
                  role='Failure diagnosis only. Checkpoint selection uses 200-row teacher-forced validation loss. No test results or test-based tuning.',
                  limitations='Small fixed-order subset, not the final test. Pixel distance does not establish instruction following. Retain failures with identity fallback.')
    dest = ROOT / 'outputs/evaluation-v1/validation-diagnostics.json'
    dest.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(path=str(dest), pending=pending,
                         models={m:{k:s[k] for k in ['n','coverage','mean_baseline_l1','mean_l1','improved_count','worsened_count','length_limit_count']} for m,s in summaries.items()}),indent=2))


if __name__ == '__main__':
    main()

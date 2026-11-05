"""Execute known validation labels to diagnose target/editor/reference mismatch.

This does not run a language model, choose parameters or change model selection.
It uses exactly the same first 32 validation IDs as the generation diagnostics.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluation_core import Scorer, parse_parameters
from summarize_evaluation import paired_summary

ROOT = Path('/mnt/sdc/zhc/IEA')


def main():
    protocol_path = ROOT / 'configs/evaluation-protocol-v1.json'
    protocol = json.loads(protocol_path.read_text())
    source = Path(protocol['split_directory']) / 'validation.json'
    manifest = json.loads(source.with_name('manifest.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['files']['validation']['sha256']
    rows = json.loads(source.read_text())[:32]
    scorer = Scorer(ROOT, protocol)
    records = []
    for row in rows:
        parsed = parse_parameters('```json\n' + json.dumps(row['extra_info']['label']) + '\n```', protocol)
        assert parsed['valid']
        result = scorer.score(row, parsed, None)
        assert result['editor_success'] and result['input_readable'] and result['reference_readable']
        records.append(dict(id=row['id'], group_id=row['group_id'], parameters=parsed['parameters'], scoring=result))
    tolerance = protocol['metrics']['comparison_tolerance']
    changes = [r['scoring']['l1_improvement'] for r in records]
    summary = dict(created_at=datetime.now(timezone.utc).isoformat(), split='validation', sample_count=len(records),
                   role='Distilled target parameters through the fixed editor: label/reference diagnostic only, not a predictive model or upper bound.',
                   mean_original_l1=float(np.mean([r['scoring']['baseline_metrics']['pixel_L1'] for r in records])),
                   mean_target_parameter_l1=float(np.mean([r['scoring']['edited_metrics']['pixel_L1'] for r in records])),
                   improved_count=sum(v > tolerance for v in changes), worsened_count=sum(v < -tolerance for v in changes),
                   unchanged_count=sum(abs(v) <= tolerance for v in changes), l1_improvement=paired_summary(changes),
                   records=records)
    output = ROOT / 'outputs/evaluation-v1/validation-target-diagnostics.json'
    assert not output.exists(), 'Preserve the existing diagnostic.'
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()

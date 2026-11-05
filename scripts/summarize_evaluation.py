"""Aggregate every frozen test row, with paired uncertainty and explicit failures."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/sdc/zhc/IEA')


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def paired_summary(differences, draws=10000, seed=20261031):
    data = np.asarray(differences, dtype=np.float64)
    if not len(data):
        return dict(n=0, mean=None, ci95=None)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), size=(draws, len(data)))].mean(axis=1)
    return dict(n=len(data), mean=float(data.mean()), ci95=[float(x) for x in np.quantile(means, [.025, .975])])


def summarize_model(rows, tolerance):
    n = len(rows)
    if not n:
        return dict(n=0)
    for row in rows:
        assert row['scoring']['input_readable'] and row['scoring']['reference_readable'], row['id']
        assert not row['scoring']['metric_error'], row['id']
    successful = [row for row in rows if row['scoring']['editor_success']]
    def mean_metric(selected, field, metric):
        return float(np.mean([row['scoring'][field][metric] for row in selected])) if selected else None
    improvement = [row['scoring']['l1_improvement'] for row in rows]
    parsed = [row for row in rows if row['parsing']['valid']]
    tp = sum(row['scoring']['parameter_metrics']['nonzero_true_positive'] for row in parsed)
    pred = sum(row['scoring']['parameter_metrics']['nonzero_predicted'] for row in parsed)
    target_valid = sum(row['scoring']['parameter_metrics']['nonzero_target'] for row in parsed)
    target_all = sum(sum(v != 0 for v in row['target_parameters'].values()) for row in rows)
    return dict(n=n, generation_success_count=sum(row['inference_error'] is None for row in rows),
                json_valid_count=sum(row['parsing']['json_valid'] for row in rows),
                tool_names_valid_count=sum(row['parsing']['tool_names_valid'] for row in rows),
                values_valid_count=sum(row['parsing']['values_valid'] for row in rows),
                editor_success_count=len(successful), coverage=len(successful)/n,
                length_limit_count=sum(row['finish_reason'] == 'length' for row in rows),
                mean_baseline_l1=mean_metric(rows, 'baseline_metrics', 'pixel_L1'),
                mean_l1=mean_metric(rows, 'edited_metrics', 'pixel_L1'),
                mean_rmse=mean_metric(rows, 'edited_metrics', 'pixel_L2'),
                mean_project_score=mean_metric(rows, 'edited_metrics', 'final_score'),
                successful_only_mean_l1=mean_metric(successful, 'edited_metrics', 'pixel_L1'),
                successful_only_mean_rmse=mean_metric(successful, 'edited_metrics', 'pixel_L2'),
                improved_count=sum(x > tolerance for x in improvement),
                worsened_count=sum(x < -tolerance for x in improvement),
                unchanged_count=sum(abs(x) <= tolerance for x in improvement),
                l1_improvement_vs_original=paired_summary(improvement),
                parse_errors=dict(Counter(row['parsing']['error'] for row in rows if row['parsing']['error'])),
                editor_errors=dict(Counter(row['scoring']['editor_error'] for row in rows if row['scoring']['editor_error'])),
                inference_errors=dict(Counter(row['inference_error'] for row in rows if row['inference_error'])),
                total_generated_tokens=sum(row['response_length'] for row in rows),
                total_generation_seconds=sum(row['generation_seconds'] for row in rows),
                tool_precision_valid_predictions=tp/pred if pred else None,
                tool_recall_valid_rows=tp/target_valid if target_valid else None,
                tool_recall_all_rows_with_failures_as_no_tools=tp/target_all if target_all else None,
                parameter_mae16_valid_rows=float(np.mean([row['scoring']['parameter_metrics']['mean_absolute_error_16'] for row in parsed])) if parsed else None)


def main():
    protocol_path = ROOT / 'configs/evaluation-protocol-v1.json'
    p = read(protocol_path)
    assert p['status'] == 'frozen'
    lock = read(ROOT / 'configs/evaluation-models-v1.json')
    assert lock['protocol_sha256'] == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    test = read(Path(p['split_directory']) / 'test.json')
    expected = [r['id'] for r in test]
    assert len(test) == len(set(expected)) == len({r['group_id'] for r in test}) == 200
    output = ROOT / 'outputs/evaluation-v1'
    datasets = {}
    for model in p['models']:
        folder = output / f'test-{model}-full'
        status = read(folder / 'status.json')
        assert status['status'] == 'completed' and status['completed'] == 200
        run = read(folder / 'run.json')
        assert run['identity']['protocol_sha256'] == lock['protocol_sha256']
        rows = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
        assert [row['id'] for row in rows] == expected
        assert [row['group_id'] for row in rows] == [row['group_id'] for row in test]
        datasets[model] = rows
    baseline = [r['scoring']['baseline_metrics'] for r in datasets['Base']]
    for model, rows in datasets.items():
        assert [r['scoring']['baseline_metrics'] for r in rows] == baseline, model
    summaries = {model: summarize_model(rows, p['metrics']['comparison_tolerance']) for model, rows in datasets.items()}
    rare = {model: summarize_model([r for r in rows if r['rare_tool_subgroup']], p['metrics']['comparison_tolerance']) for model, rows in datasets.items()}
    from evaluation_core import Scorer, parse_parameters
    scorer = Scorer(ROOT, p)
    target_control = []
    for row in test:
        parsed = parse_parameters('```json\n' + json.dumps(row['extra_info']['label']) + '\n```', p)
        assert parsed['valid'], row['id']
        result = scorer.score(row, parsed, None)
        assert result['editor_success'] and result['baseline_metrics'] == baseline[len(target_control)]
        target_control.append(result)
    target_summary = dict(
        interpretation='Execute the distilled target parameters through the same editor. Diagnostic label-quality control only, not a predictive model or a theoretical upper bound.',
        mean_l1=float(np.mean([r['edited_metrics']['pixel_L1'] for r in target_control])),
        mean_rmse=float(np.mean([r['edited_metrics']['pixel_L2'] for r in target_control])),
        l1_improvement_vs_original=paired_summary([r['l1_improvement'] for r in target_control]))
    comparisons = {}
    for better, reference in [('A', 'Base'), ('B', 'A'), ('A', 'BaseSchema'), ('B', 'BaseSchema'), ('Official', 'A')]:
        differences = [ref['scoring']['edited_metrics']['pixel_L1'] - candidate['scoring']['edited_metrics']['pixel_L1']
                       for candidate, ref in zip(datasets[better], datasets[reference])]
        comparisons[f'{better}_minus_{reference}'] = dict(
            positive_means=f'{better} has lower pixel L1 than {reference}',
            all_rows=paired_summary(differences),
            rare_tool_rows=paired_summary([d for d, row in zip(differences, test)
                                          if any(row['extra_info']['label'].get(k, 0) != 0 for k in ('exposure', 'sharpness'))]))
    summary = dict(created_at=datetime.now(timezone.utc).isoformat(), protocol_sha256=lock['protocol_sha256'],
                   sample_count=200, independent_source_groups=200, models=summaries, rare_tool_subgroup=rare,
                   comparisons=comparisons, distilled_target_parameter_control=target_summary,
                   failure_policy=p['metrics']['failure_policy'],
                   bootstrap='Paired percentile bootstrap of the 200 source groups, 10000 draws, seed 20261031; exploratory intervals, not multiplicity-adjusted.',
                   official_scope=p['official_scope'], supplementary_control=p['supplementary_control'])
    write(output / 'summary.json', summary)
    fields = ['model', 'n', 'editor_success_count', 'coverage', 'mean_l1', 'mean_rmse', 'mean_project_score',
              'successful_only_mean_l1', 'improved_count', 'worsened_count', 'unchanged_count', 'length_limit_count']
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model, result in summaries.items():
            writer.writerow(dict(model=model, **{key: result[key] for key in fields if key != 'model'}))
    # Complete per-row comparison supports failure audit without selecting favorable examples.
    with (output / 'paired-rows.jsonl').open('w') as stream:
        for i, row in enumerate(test):
            comparison = dict(id=row['id'], group_id=row['group_id'], instruction=row['extra_info']['instruction'],
                              baseline=baseline[i], distilled_target_control=target_control[i],
                              models={m: dict(parsing=rs[i]['parsing'], scoring=rs[i]['scoring']) for m, rs in datasets.items()})
            stream.write(json.dumps(comparison, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

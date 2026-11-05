"""Plot only verified training summaries and complete frozen test statistics."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from audit_panel_alignment import require_matplotlib_panel_alignment


def save(fig, stem):
    require_matplotlib_panel_alignment(fig, json_out=str(stem)+'.alignment.json',
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(str(stem)+'.pdf')
    fig.savefig(str(stem)+'.svg')
    fig.savefig(str(stem)+'.png', dpi=300)
    plt.close(fig)


def draw_interval(ax, y, result, color):
    mean, bounds = result['mean'], result['ci95']
    assert result['n'] > 0 and all(np.isfinite([mean, *bounds]))
    ax.errorbar(mean, y, xerr=[[mean-bounds[0]], [bounds[1]-mean]],
                fmt='o', color=color, capsize=2, markersize=4, linewidth=1)


def limits_from_intervals(results):
    bound = max(abs(x) for result in results for x in [result['mean'], *result['ci95']])
    return (-max(bound*1.2, .001), max(bound*1.2, .001))


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--root', type=Path, default=Path('/mnt/sdc/zhc/IEA'))
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--training-only', action='store_true')
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    colors = {'Base': '#777777', 'BaseSchema': '#A0A0A0', 'A': '#366D9B',
              'B': '#BE7D47', 'Official': '#638C82'}
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'font.size': 7, 'axes.titlesize': 8, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': .7})
    if args.training_only:
        plot_training(args.root, args.output, colors)
        print(json.dumps(dict(output=str(args.output), role='verified_training_curves_only')))
        return
    summary_path = args.root / 'outputs/evaluation-v1/summary.json'
    results = json.loads(summary_path.read_text())
    assert results['sample_count'] == results['independent_source_groups'] == 200
    models = ['Base', 'BaseSchema', 'A', 'B', 'Official']
    assert set(models) == set(results['models'])
    assert all(results['models'][m]['n'] == 200 for m in models)
    fig, axes = plt.subplots(1, 3, figsize=(183/25.4, 90/25.4))
    fig.subplots_adjust(left=.12, right=.985, top=.82, bottom=.31, wspace=.65)
    y = np.arange(len(models))
    ax = axes[0]
    ax.barh(y, [100*results['models'][m]['coverage'] for m in models],
            color=[colors[m] for m in models], height=.55)
    ax.set(yticks=y, yticklabels=models, xlim=(0, 105), xticks=[0, 50, 100],
           xlabel='Executable outputs (%)', title='a  Execution coverage')
    ax.set_ylim(4.6, -.6)
    ax = axes[1]
    intervals = [results['models'][m]['l1_improvement_vs_original'] for m in models]
    for pos, model, interval in zip(y, models, intervals):
        draw_interval(ax, pos, interval, colors[model])
    ax.axvline(0, color='#BBBBBB', linestyle='--', linewidth=.7, zorder=0)
    ax.set(yticks=y, yticklabels=[], xlabel='Original L1 - model L1',
           title='b  Change vs original', xlim=limits_from_intervals(intervals))
    ax.set_ylim(4.6, -.6)
    ax = axes[2]
    comparison = results['comparisons']['B_minus_A']
    intervals = [comparison['all_rows'], comparison['rare_tool_rows']]
    for pos, interval in enumerate(intervals):
        draw_interval(ax, pos, interval, colors['B'])
    ax.axvline(0, color='#BBBBBB', linestyle='--', linewidth=.7, zorder=0)
    ax.set(yticks=[0, 1], yticklabels=['All', 'Rare tools'],
           xlabel='A L1 - B L1', title='c  Effect of resampling',
           xlim=limits_from_intervals(intervals), ylim=(1.6, -.6))
    fig.text(.12, .96, 'Frozen held-out evaluation: output validity and paired editing changes',
             fontsize=9, fontweight='bold', va='top')
    rare_n = comparison['rare_tool_rows']['n']
    fig.text(.12, .17, f'200 source groups; rare-tool subgroup n = {rare_n}. Positive differences mean lower pixel L1.\n'
             'Points: mean paired change; bars: 95% bootstrap intervals (10,000 draws). One training seed.\n'
             'Failures use the original image. BaseSchema changes the prompt; Official is a reference only.',
             fontsize=7, linespacing=1.5, va='top')
    save(fig, args.output / 'held-out-comparison')
    plot_training(args.root, args.output, colors)
    # Preserve all statistics used by the plots, including failure/coverage details.
    (args.output / 'test-source-data.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), test_rows=200, models=models)))


def plot_training(root, output, colors):
    training = {}
    for model in ['A', 'B']:
        source = root / f'logs/stage1-{model}-3k/summary.json'
        summary = json.loads(source.read_text())
        assert summary['status'] == 'completed_and_reload_verified' and summary['global_step'] == 376
        training[model] = summary
    fig, ax = plt.subplots(figsize=(120/25.4, 80/25.4))
    fig.subplots_adjust(left=.17, right=.96, bottom=.25, top=.86)
    for model, summary in training.items():
        history = summary['validation_history']
        assert [r['step'] for r in history] == [94, 188, 282, 376]
        ax.plot([r['step'] for r in history], [r['eval_loss'] for r in history],
                marker='o' if model == 'A' else 's', linestyle='-' if model == 'A' else '--',
                markersize=4, color=colors[model], label=model, linewidth=1.3)
    ax.set(xlabel='Optimizer update', ylabel='Validation cross-entropy loss',
           xticks=[94, 188, 282, 376], title='Validation-based checkpoint selection')
    ax.legend(loc='upper right')
    fig.text(.17, .06, '200 fixed validation rows; one seed per model. No test-set selection.', fontsize=7)
    save(fig, output / 'validation-curves')
    with (output / 'training-source-data.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model', 'step', 'eval_loss'])
        writer.writeheader()
        for model, summary in training.items():
            for point in summary['validation_history']:
                writer.writerow(dict(model=model, step=point['step'], eval_loss=point['eval_loss']))


if __name__ == '__main__':
    main()

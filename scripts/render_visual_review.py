"""Render every selected case with one shared, full-frame image display policy."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from audit_panel_alignment import require_matplotlib_panel_alignment


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('bundle', type=Path)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    manifest = json.loads((args.bundle / 'manifest.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    panel_reviews = []
    for case in manifest['cases']:
        # 270 x 185 mm is an internal inspection sheet, not a journal-sized figure.
        fig, axes = plt.subplots(2, 4, figsize=(270 / 25.4, 185 / 25.4))
        fig.subplots_adjust(left=.025, right=.975, bottom=.12, top=.79, wspace=.08, hspace=.28)
        for ax in axes.flat:
            ax.set_axis_off()
            ax.set_visible(False)
        for ax, item in zip(axes.flat, case['images']):
            source = args.bundle / item['path']
            assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256']
            with Image.open(source) as im:
                # Only a display transform: all source files are unchanged.
                rgb = im.convert('RGB')
                fitted = ImageOps.contain(rgb, (640, 480), Image.Resampling.LANCZOS)
                canvas = Image.new('RGB', (640, 480), 'white')
                canvas.paste(fitted, ((640-fitted.width)//2, (480-fitted.height)//2))
            ax.set_visible(True)
            ax.imshow(canvas, interpolation='none')
            ax.set_title(item['label'], fontsize=9, pad=7)
            panel_reviews.append(dict(case=case['case'], panel=item['label'], source=item['path'],
                                      source_sha256=item['sha256'], crop='none',
                                      colour_adjustment='none', display='640x480 white letterbox, preserve aspect ratio'))
        label = 'TRAINING LAYOUT CHECK' if manifest['role'] == 'training_layout_preview' else 'Blinded test review'
        fig.text(.025, .96, f"{label}  |  Case {case['case']:02d}  |  {case['id']}",
                 fontsize=11, fontweight='bold', va='top')
        wrapped = textwrap.fill('Instruction: ' + ' '.join(case['instruction'].split()), width=130)
        fig.text(.025, .90, wrapped, fontsize=9, va='top', linespacing=1.4)
        fig.text(.025, .055, 'Full frames; identical display scale and letterboxing; no colour or contrast adjustments.\n'
                 'Inspect instruction direction, adjustment strength and unrelated changes. Model numbers vary by case.',
                 fontsize=8, va='bottom', linespacing=1.4)
        base = args.output / f"case-{case['case']:02d}"
        require_matplotlib_panel_alignment(fig, json_out=str(base)+'.alignment.json',
                                           tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
        fig.savefig(str(base)+'.pdf')
        fig.savefig(str(base)+'.svg')
        fig.savefig(str(base)+'.png', dpi=300)
        plt.close(fig)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), source_manifest=str(args.bundle / 'manifest.json'),
                  case_count=len(manifest['cases']), panels=panel_reviews,
                  note='Alignment gate passed before export; PDF text/collision audits and direct visual review remain required.')
    (args.output / 'render-provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), cases=len(manifest['cases']))))


if __name__ == '__main__':
    main()

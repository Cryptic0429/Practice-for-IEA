"""Audit all 30 blinded sheets and both quantitative figures, without dropping failures."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/mnt/sdc/zhc/IEA')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(tool, args):
    command = [sys.executable, str(ROOT / 'scripts' / tool), *map(str, args), '--json']
    run = subprocess.run(command, capture_output=True, text=True)
    try:
        report = json.loads(run.stdout)
    except json.JSONDecodeError:
        report = dict(error='Audit did not return valid JSON', stdout=run.stdout, stderr=run.stderr)
    return dict(command=command, returncode=run.returncode, report=report)


def main():
    folder = ROOT / 'outputs/evaluation-v1'
    expected = [folder / 'blinded-visual-render' / f'case-{i:02d}.pdf' for i in range(1, 31)]
    expected += [folder / 'result-figures' / f'{name}.pdf' for name in ['held-out-comparison', 'validation-curves']]
    assert all(path.is_file() for path in expected), 'All 32 final PDFs are required.'
    assert len(list((folder / 'blinded-visual-render').glob('case-*.pdf'))) == 30
    sources = {}
    for script in ['render_visual_review.py', 'plot_experiment_results.py']:
        path = ROOT / 'scripts' / script
        result = check('validate_figure.py', [path])
        sources[script] = dict(sha256=sha(path), **result)
    figures = []
    for pdf in expected:
        alignment_path = pdf.with_suffix('.alignment.json')
        alignment = json.loads(alignment_path.read_text())
        font = check('audit_pdf_text.py', [pdf, '--min-pt', '5'])
        collision = check('audit_figure_collisions.py', [pdf])
        for suffix, result in [('text-audit', font), ('collision-audit', collision)]:
            pdf.with_suffix('.' + suffix + '.json').write_text(json.dumps(result, indent=2) + '\n')
        fresh_alignment = 0 <= pdf.stat().st_mtime - alignment_path.stat().st_mtime < 60
        passed = (fresh_alignment and alignment['verdict'] in ['PASS', 'NOT APPLICABLE']
                  and font['returncode'] == 0 and collision['returncode'] == 0
                  and collision['report'].get('verdict') == 'PASS')
        figures.append(dict(path=str(pdf), sha256=sha(pdf), png_sha256=sha(pdf.with_suffix('.png')),
                            alignment_path=str(alignment_path), alignment_sha256=sha(alignment_path),
                            alignment_fresh=fresh_alignment, alignment_verdict=alignment['verdict'],
                            font_audit=font, collision_audit=collision, passed=passed))
    result = dict(audited_at=datetime.now(timezone.utc).isoformat(), expected_pdf_count=32,
                  source_preflight=sources, figures=figures,
                  automated_checks_passed=all(item['passed'] for item in figures)
                      and all(item['returncode'] == 0 for item in sources.values()),
                  source_warnings_reviewed=[
                      'PNG 300 dpi is the internal-report preview; editable PDF and SVG are supplied. No journal TIFF submission is being claimed.',
                      '270 mm visual inspection sheets and 120 mm training plots intentionally differ from journal column defaults.'
                  ],
                  scope='Automated layout/text checks only. Direct inspection of each panel and scientific interpretation are separate required acceptance evidence.')
    (folder / 'figure-audit-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(pdf_count=32, automated_checks_passed=result['automated_checks_passed'],
                         failed_figures=[r['path'] for r in figures if not r['passed']]), indent=2))
    if not result['automated_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

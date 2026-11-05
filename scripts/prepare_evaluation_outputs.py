"""After five full tests finish, prepare statistics and concealed-label review sheets.

CPU-only; never performs model inference, decodes the blinding key for the
reviewer, records visual judgments, or declares final acceptance.
"""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/mnt/sdc/zhc/IEA')
LOG = ROOT / 'logs/evaluation-v1'
OUT = ROOT / 'outputs/evaluation-v1'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def write_status(state):
    path = LOG / 'preparation-status.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(path)


def main():
    lock = (LOG / 'preparation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='waiting_for_all_tests', pid=os.getpid(), started_at=stamp(), steps=[])
    assert not (OUT / 'summary.json').exists(), 'Preserve previous summaries.'
    try:
        while True:
            queue = read(LOG / 'queue-status.json')
            assert queue['status'] != 'attention_required', queue.get('error')
            state.update(updated_at=stamp(), queue_status=queue['status'])
            write_status(state)
            if queue['status'] == 'completed':
                break
            time.sleep(10)
        for model in ['Base', 'A', 'B', 'Official', 'BaseSchema']:
            status = read(OUT / f'test-{model}-full/status.json')
            assert status['status'] == 'completed' and status['completed'] == status['total'] == 200
        # Child environments expose no GPUs, including the protected Demo card.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1')
        tasks = [
            ('statistics', 'iea', 'summarize_evaluation.py', []),
            ('blinded_bundle', 'iea', 'export_visual_bundle.py', []),
            ('blinded_sheets', 'report', 'render_visual_review.py',
             [str(OUT / 'blinded-visual-bundle'), '--output', str(OUT / 'blinded-visual-render')]),
            ('result_figures', 'report', 'plot_experiment_results.py',
             ['--output', str(OUT / 'result-figures')]),
            ('figure_audits', 'report', 'audit_result_figures.py', []),
            ('blank_observations', 'iea', 'visual_review_records.py', ['prepare']),
        ]
        for stage, environment, script, args in tasks:
            state.update(status='preparing', current_stage=stage, updated_at=stamp())
            write_status(state)
            command = [str(ROOT / f'envs/{environment}/bin/python'), str(ROOT / 'scripts' / script), *args]
            started = stamp()
            with (LOG / f'prepare-{stage}.log').open('xb') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            state['steps'].append(dict(stage=stage, started_at=started, finished_at=stamp(), command=command))
        state.update(status='ready_for_visual_review', updated_at=stamp(), finished_at=stamp(),
                     next_action='Inspect all 30 concealed-label sheets, save observations, then decode and write final report.')
        write_status(state)
    except BaseException as exc:
        state.update(status='attention_required', updated_at=stamp(), error=f'{type(exc).__name__}: {exc}')
        write_status(state)
        raise


if __name__ == '__main__':
    main()

"""Write the final local receipt after archive verification and monitor shutdown."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tomllib


def main():
    root = Path(__file__).resolve().parents[1]
    meta = root / 'metadata/stage2-A-edit-grpo-v1'
    delivery = root / 'artifacts/stage2-acceptance-v1'
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    local = read(delivery / 'local-download-verification.json')
    remote = read(delivery / 'archive-verification.json')
    assert local['all_member_hashes_verified'] and remote['all_member_hashes_verified']
    for key in ('sha256', 'bytes', 'file_count'):
        assert local[key] == remote[key]
    archive = delivery / 'iea-stage2-acceptance-v1.tar.gz'
    assert archive.stat().st_size == local['bytes']
    manifest = read(delivery / 'MANIFEST.json')
    entries = {e['path']: e for e in manifest['files']}
    assert len(entries) == local['file_count']
    prearchive = read(meta / 'final-evidence/prearchive-acceptance.json')
    assert prearchive['status'] == 'ready_to_archive'
    for value in prearchive['requirements'].values():
        assert value['passed']
        for p, digest in value['evidence'].items():
            assert entries[p]['sha256'] == digest
    numerical = read(meta / 'final-evidence/closeout-numerical-audit.json')
    for job in numerical['jobs'].values():
        for image in job['saved_images']:
            p = str(Path(image['path']).relative_to('/mnt/sdc/zhc/IEA'))
            assert entries[p]['sha256'] == image['sha256']
    for folder in ('formal-v1', 'sft-control'):
        for step in (64, 128):
            for name in ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json'):
                assert f'outputs/stage2-A-edit-grpo-v1/{folder}/checkpoint-{step}/{name}' in entries
    for doc in ('README.md', 'Stage2_GRPO_方案与执行说明.md', 'IEA_SJTU1_两天任务表与规划.md', '验收_Stage2_完整实验报告.md'):
        assert sha(root / doc) == entries[doc]['sha256']
    automation = Path.home() / '.codex/automations/sjtu1-stage-2-15/automation.toml'
    settings = tomllib.loads(automation.read_text())
    assert settings['status'] == 'PAUSED', 'Stop completed monitoring before final receipt'
    monitor = dict(id=settings['id'], status=settings['status'], updated_at=settings['updated_at'],
                   reason='Full experiment and download verification complete; no more scheduled polling needed')
    now = datetime.now(timezone.utc).isoformat()
    requirements = dict(prearchive['requirements'])
    requirements['archive_and_local_download'] = dict(passed=True, files=local['file_count'],
                                                       archive_sha256=local['sha256'],
                                                       local_report_sha256=sha(delivery / 'local-download-verification.json'))
    requirements['finish_monitoring'] = dict(passed=True, automation=monitor)
    assert len(requirements) == 15 and all(v['passed'] for v in requirements.values())
    receipt = dict(status='complete', completed_at=now, requirements=requirements,
                   archive=local, manifest_sha256=sha(delivery / 'MANIFEST.json'),
                   prearchive_sha256=sha(meta / 'final-evidence/prearchive-acceptance.json'),
                   total_gpu_hours=prearchive['gpu_hours'], numerical_predictions=900,
                   validation_predictions=400, anonymous_observations=90,
                   result='No demonstrated new-held-out-test benefit from GRPO. Old regression is exploratory.',
                   external_assets=manifest['external_assets'], limits=prearchive['limits'],
                   note='This receipt and monitor shutdown occur after the immutable archive snapshot; packaged historical progress files retain their original timestamps.')
    with (delivery / 'final-acceptance.json').open('x') as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
    (meta / 'final-acceptance.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    (meta / 'monitor-completion.json').write_text(json.dumps(monitor, ensure_ascii=False, indent=2) + '\n')
    progress = read(meta / 'progress.json')
    progress.update(status='complete', updated_at=now, remaining_work=[],
                    final_acceptance='artifacts/stage2-acceptance-v1/final-acceptance.json',
                    last_goal_turn_classification='complete: full fifteen-item acceptance, archive downloaded and every member verified')
    (meta / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2) + '\n')
    mapping = read(meta / 'completion-requirements.json')
    mapping.update(status='complete', updated_at=now)
    for name, value in requirements.items():
        mapping['requirements'][name].update(status='passed', final_evidence=value)
    (meta / 'completion-requirements.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + '\n')
    events = read(meta / 'reported-events.json')
    if not any(e['id'] == 'stage2_full_acceptance_complete' for e in events['events']):
        events['events'].append(dict(id='stage2_full_acceptance_complete', reported_at=now,
                                     receipt='artifacts/stage2-acceptance-v1/final-acceptance.json', archive_sha256=local['sha256']))
    (meta / 'reported-events.json').write_text(json.dumps(events, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status='complete', requirements=15, files=local['file_count'],
                          archive_sha256=local['sha256'], total_gpu_hours=prearchive['gpu_hours'])))


if __name__ == '__main__':
    main()

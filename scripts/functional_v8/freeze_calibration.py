"""Freeze calibration only after the new opposite renders were actually viewed."""
import json
from pathlib import Path

from functional_v1.common import ROOT, read, write, sha, stamp
from .prepare_calibration import EXP, DATA, REWARD_SHA, REWARD_REVISION, CODE_REVISION
from .calibrate_reward import prompts, tasks, encode


def main():
    from transformers import AutoProcessor
    from functional_v1.common import BASE
    destination = EXP / 'calibration-protocol.json'
    assert not destination.exists()
    preparation = read(EXP / 'calibration-preparation.json')
    cases = read(DATA / 'calibration-candidates.json')
    notes_path = EXP / 'calibration-direct-observations.jsonl'
    notes = list(map(json.loads, notes_path.read_text().splitlines()))
    assert len(notes) == 10 and {r['id'] for r in notes} == {r['id'] for r in cases['direction']}
    for note in notes:
        assert note['good'] == {'completion': 'completed', 'quality': 'acceptable'}
        assert note['opposite']['completion'] == 'failed'
        assert sha(ROOT / note['sheet']['sheet']) == note['sheet']['sha256']
    adapter = ROOT / 'models/external/EditScore-7B'
    assert sha(adapter / 'adapter_model.safetensors') == REWARD_SHA
    assert read(adapter / 'adapter_config.json')['base_model_name_or_path'].endswith('/cc594898137f460bfe9f0759e9844b3ce807cfb5')
    processor = AutoProcessor.from_pretrained(BASE, max_pixels=262144)
    token_audit = []
    for task in tasks():
        batch = encode(processor, task)
        token_audit.append(dict(id=task['id'], input_tokens=batch['input_ids'].shape[1], images=len(task['images'])))
    assert len(token_audit) == 76
    write(EXP / 'input-token-audit.json', dict(at=stamp(), audit=token_audit,
        input_fields=['only input/edited image pixels and official prompt with one language instruction; PQ uses only the edited image'],
        excluded=['AI grades', 'review notes', 'expected rank', 'role names', 'model identity', 'parameters', 'reference targets outside the candidate being scored']))
    files = [ROOT / 'scripts/functional_v8' / name for name in ('prepare_calibration.py', 'calibrate_reward.py', 'freeze_calibration.py')]
    files += [ROOT / 'third_party/EditScore/editscore/vie_prompts.py', adapter / 'adapter_config.json']
    write(destination, dict(at=stamp(), status='frozen_before_any_reward_inference',
        dataset_sha256=sha(DATA / 'calibration-candidates.json'), direct_review_sha256=sha(notes_path),
        token_audit_sha256=sha(EXP / 'input-token-audit.json'), code_hashes={str(p.relative_to(ROOT)): sha(p) for p in files},
        external_model=dict(repository='EditScore/EditScore-7B', revision=REWARD_REVISION, adapter_sha256=REWARD_SHA,
                            source_commit=CODE_REVISION, base_revision='cc594898137f460bfe9f0759e9844b3ce807cfb5'),
        compared_routes=['base', 'editscore'], precision='Full BF16, adapter merged after exact tensor verification',
        sampling=dict(seed=20261114, temperature=.7, top_p=.9, top_k=20, passes=1, max_new_tokens=512),
        official_prompt_unchanged=True, official_manual_img_tag_template=True,
        implementation_differences=['262144 pixel limit per image', 'sdpa attention',
                                   'No JSON repair, retries, guessed scores or default values; invalid outputs fail calibration'],
        gates=preparation['initial_gates'], GPU_limit_seconds_each=1800, shared_original_GPU_hour_cap=16,
        scope=preparation['data_scope'], limitations=preparation['limitations'],
        quality_calibration='Eight minor-defect comparisons only, not certification for major-damage detection; some PQ images answer different requests, and PQ has no instruction input.',
        own_policy_training_started=False, acceptance_opened=False,
        posthoc_policy='Do not modify threshold, cases, prompt or parsing after seeing results. A failure stops use of this configuration as a training reward. Report all errors and ties.'))
    print('Frozen18 training-photo groups9/9,10 bilingual direction tasks,8 quality pairs;76 inputs audited; noGPU inference started')


if __name__ == '__main__':
    main()

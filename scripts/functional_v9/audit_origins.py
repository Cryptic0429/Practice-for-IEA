"""CPU identities and input compatibility before any round9 model forward."""
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoProcessor

from functional_v1.common import ROOT, BASE, read, write, sha, stamp
from functional_v1.runtime import processor, encode, image_input, messages
from functional_v7.outcomes import input_row, target_text

EXP = ROOT / 'experiments/functional-v9-20261106'
DATA = ROOT / 'data/training/functional-v9-20261106'
ORIGINS = {'base': BASE, 'author': ROOT / 'models/Qwen2.5-VL-7B-IEA'}


def stream_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    destination = EXP / 'origin-audit.json'
    assert not destination.exists()
    torch.set_num_threads(4)
    prior_file = ROOT / 'configs/evaluation-weight-integrity-v1.json'
    prior = read(prior_file)
    rows = read(ROOT / 'data/training/functional-v7-20261105/train-accepted.json')
    cases = {c['id']: c for c in read(ROOT / 'data/training/functional-v7-20261105/cases.json')}
    assert len(rows) == 924
    identities = {}
    for route, path in ORIGINS.items():
        old = prior['models']['Base' if route == 'base' else 'Official']
        expected = {Path(s['path']).name: s['sha256'] for s in old['shards']}
        index = read(path / 'model.safetensors.index.json')
        shards = []
        for name in sorted(set(index['weight_map'].values())):
            file = path / name
            before = file.stat(); digest = stream_sha(file); after = file.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            assert digest == expected[name]
            shards.append(dict(path=str(file.relative_to(ROOT)), sha256=digest, bytes=after.st_size, mtime_ns=after.st_mtime_ns))
        config = AutoConfig.from_pretrained(path)
        text = config.text_config
        identities[route] = dict(path=str(path.relative_to(ROOT)), shards=shards,
            architecture={name: getattr(text, name) for name in ('hidden_size', 'intermediate_size', 'num_hidden_layers',
                         'num_attention_heads', 'num_key_value_heads', 'vocab_size')},
            assets={file.name: sha(file) for file in path.iterdir() if file.is_file() and file.suffix in ('.json', '.jinja') and file.stat().st_size < 20_000_000},
            prior_integrity_file=str(prior_file.relative_to(ROOT)), prior_integrity_sha256=sha(prior_file),
            source='Pinned original Qwen revision cc594898137f460bfe9f0759e9844b3ce807cfb5' if route == 'base'
                   else 'Existing author IEA full weights; publisher checksum not independently attested; exact content matches historical integrity receipt')
    assert identities['base']['architecture'] == identities['author']['architecture']
    assert [s['sha256'] for s in identities['base']['shards']] != [s['sha256'] for s in identities['author']['shards']]
    shared = processor()
    author = AutoProcessor.from_pretrained(ORIGINS['author'], use_fast=True)
    author.tokenizer.padding_side = 'left'; author.image_processor.max_pixels = 262144
    tokenizer_records = []
    for row in rows:
        case = cases[row['case_id']]
        value = input_row(case, row['language'])
        rendered = shared.apply_chat_template(messages(value) + [{'role': 'assistant', 'content': target_text(row['decision'])}], tokenize=False, add_generation_prompt=False)
        left = shared.tokenizer(rendered, add_special_tokens=False)['input_ids']
        right = author.tokenizer(rendered, add_special_tokens=False)['input_ids']
        assert left == right, row['id']
        tokenizer_records.append(dict(id=row['id'], tokens=len(left), shared_serialized_text_token_ids_identical=True))
    probes = []
    # Training photos only. The exact shared base chat template is deliberately used for both origins.
    for row in rows[:8]:
        case = cases[row['case_id']]; value = input_row(case, row['language'])
        before = encode(shared, value)
        serialized = shared.apply_chat_template(messages(value), tokenize=False, add_generation_prompt=True)
        after = author(text=[serialized], images=[image_input(value['images'][0])], return_tensors='pt', add_special_tokens=False)
        equal = set(before) == set(after) and all(torch.equal(before[key], after[key]) for key in before)
        probes.append(dict(id=row['id'], all_input_tensors_identical_under_shared_template=equal,
                           keys={key: list(v.shape) for key, v in before.items()}))
    assert all(p['all_input_tensors_identical_under_shared_template'] for p in probes)
    write(destination, dict(at=stamp(), status='CPU_identity_and_shared_input_compatibility_passed', origins=identities,
        tokenizer_records=tokenizer_records, input_tensor_probes=probes, checked_training_rows=924,
        chosen_processor='Original Qwen processor/chat template for BOTH routes, exactly as round7. Author vocabulary IDs verified identical on all924 serialized training inputs and assistant targets. Asset files are not assumed byte-identical.',
        direct_model_forward_calls=0, GPU_hours=0, policy_training_started=False,
        author_overlap_caveat='Historical source audit confirmed original-image overlap in author train/test splits. Author may have seen heldout GIER sources. This origin is a practical reuse experiment, NOT a clean causal or independent generalization control.',
        original_author_GPU1_service_untouched=True))
    write(DATA / 'data-reuse.json', dict(at=stamp(), actual_training_source='data/training/functional-v7-20261105/train-accepted.json',
        dataset_sha256=sha(ROOT / 'data/training/functional-v7-20261105/train-accepted.json'),
        cases_sha256=sha(ROOT / 'data/training/functional-v7-20261105/cases.json'),
        semantic_cases=462, unique_language_rows=924, original_photo_groups=160, photo_domains={'portrait': 80, 'landscape': 80},
        outcome_counts={'edit': 379, 'keep': 60, 'decline': 23},
        new_labels_added=False, existing_admission='data/training/functional-v7-20261105/admission.json',
        note='Reuse exact admitted instruction/target decisions and editor, no resampling or acceptance access. This is a new initialization comparison, not a newly collected dataset.'))
    print('Two full-weight origins match recorded SHA identities;924 token inputs and8 image probes match; zero GPU forwards')


if __name__ == '__main__':
    main()

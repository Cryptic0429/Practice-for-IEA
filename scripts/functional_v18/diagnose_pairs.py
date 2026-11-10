"""Score crossed mild/strong labels and generate from input-only TRAIN prefixes.

Teacher-forced labels are used only for a diagnostic score. Greedy generation
gets a new image/request-only prefix and cannot see either candidate answer.
"""
import argparse
import json
import os
from pathlib import Path

from functional_v1.common import ROOT, GPUJob, read, write, append, sha, stamp

EXP = ROOT / 'experiments/functional-v18-20261107'
DATA = ROOT / 'data/training/functional-v15-20261106'
PARENT = ROOT / 'models/from-functional-v11/sft-shadow-correction/seed-01'
PARENT_SHA = '97c84f1799fb319799c86515402574b3b585593763e5388eb30d0df3e0d6ea90'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', choices=['0', '2', '3'], required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == args.gpu
    protocol = read(EXP / 'diagnostic-protocol.json')
    assert not (EXP / 'diagnostic-result.json').exists()
    assert not (EXP / 'predictions.jsonl').exists()
    for path, digest in protocol['code_hashes'].items():
        assert sha(ROOT / path) == digest, path
    assert sha(DATA / 'cases.json') == protocol['cases_sha256']
    assert sha(DATA / 'train-accepted.json') == protocol['dataset_sha256']
    assert sha(PARENT / 'adapter_model.safetensors') == PARENT_SHA
    for pair in protocol['pairs']:
        for case in pair['cases']:
            assert sha(ROOT / case['image']) == case['input_sha256']
    import torch
    import torch.nn.functional as F
    from functional_v15.runtime import load_model, processor, build_batch, predict
    from functional_v15.outcomes import execute
    from functional_v15.editor import load_rgb
    from PIL import Image

    torch.set_num_threads(4)
    output = EXP / 'images'
    output.mkdir(exist_ok=False)
    scores, predictions = [], []
    status = dict(at=stamp(), status='starting', pid=os.getpid(), gpu=args.gpu)
    write(EXP / 'status.json', status)
    with GPUJob(args.gpu, 'v18-training-pair-specificity-diagnostic', max_seconds=900) as budget:
        model = load_model('language_merger', adapter=PARENT)
        p = processor()
        for pair in protocol['pairs']:
            for side, case in enumerate(pair['cases']):
                other = pair['cases'][1-side]
                for language in ('zh', 'en'):
                    for target, decision in [('own', case['decision']), ('crossed', other['decision'])]:
                        batch = build_batch(p, case, language, decision)
                        moved = {k: v.to('cuda') for k, v in batch.items()}
                        labels = moved.pop('labels')[:, 1:]
                        mask = labels.ne(-100)
                        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                            result = model(**moved, use_cache=False)
                            logits = result.logits[:, :-1][mask].float()
                            token_logp = -F.cross_entropy(logits, labels[mask], reduction='none')
                        assert torch.isfinite(token_logp).all()
                        scores.append(dict(pair_id=pair['id'], case_id=case['id'], language=language,
                            target=target, decision=decision, assistant_tokens=int(mask.sum()),
                            sum_logp=float(token_logp.sum()), mean_logp=float(token_logp.mean())))
                        del result, logits, moved, batch, token_logp
                        budget.tick()
                # English is the predeclared generation-only diagnostic language.
                # The teacher-forced candidates above do not enter this prefix.
                response, decision = predict(model, p, case, 'en')
                execution = execute(load_rgb(ROOT / case['image']), decision)
                path = output / (case['id'] + '-en.png')
                if execution['result'] is not None:
                    Image.fromarray(execution['result']).save(path, compress_level=1)
                row = dict(pair_id=pair['id'], case_id=case['id'], language='en',
                    input_image=case['image'], input_sha256=case['input_sha256'], response=response,
                    decision=decision, status=execution['status'],
                    output_image=str(path.relative_to(ROOT)) if path.exists() else None,
                    output_sha256=sha(path) if path.exists() else None,
                    input_only_generation=True)
                predictions.append(row)
                append(EXP / 'predictions.jsonl', row)
                write(EXP / 'teacher-forcing-scores.json', scores)
                status.update(status='running', cases_completed=len(predictions), at=stamp())
                write(EXP / 'status.json', status)
                budget.tick()
        comparison = []
        for pair in protocol['pairs']:
            for case in pair['cases']:
                for language in ('zh', 'en'):
                    chosen = next(r for r in scores if (r['case_id'], r['language'], r['target']) == (case['id'], language, 'own'))
                    crossed = next(r for r in scores if (r['case_id'], r['language'], r['target']) == (case['id'], language, 'crossed'))
                    comparison.append(dict(case_id=case['id'], language=language,
                        own_minus_crossed_mean_logp=chosen['mean_logp']-crossed['mean_logp'],
                        own_minus_crossed_sum_logp=chosen['sum_logp']-crossed['sum_logp']))
    write(EXP / 'diagnostic-result.json', dict(at=stamp(),
        status='completed_training_only_diagnostic_not_a_functional_score',
        parent_sha256=PARENT_SHA, protocol_sha256=sha(EXP / 'diagnostic-protocol.json'),
        pairs=len(protocol['pairs']), semantic_inputs=len(predictions), scoring_language_rows=len(comparison),
        generated_language='en_only', comparisons=comparison,
        own_preferred_by_mean_logp=sum(r['own_minus_crossed_mean_logp'] > 0 for r in comparison),
        own_preferred_by_sum_logp=sum(r['own_minus_crossed_sum_logp'] > 0 for r in comparison),
        greedy_pair_decisions_equal=sum(predictions[i]['decision'] == predictions[i+1]['decision'] for i in range(0,len(predictions),2)),
        optimizer_steps=0, training_route_frozen=False, admitted_new_labels=0,
        development_inputs_used=False, acceptance_inputs_used=False,
        limitation='Exact parameter likelihood and unequal decisions are diagnostics only. Crossed targets are not automatically inferior. Visual review is required before any preference admission.'))
    status.update(status='completed', at=stamp(), cases_completed=len(predictions))
    write(EXP / 'status.json', status)


if __name__ == '__main__':
    main()

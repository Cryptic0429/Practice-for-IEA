"""Frozen stratified teacher-forced loss diagnosis. No acceptance data or quality scores."""
import argparse
import hashlib
from .common import ROOT, EXP, PARENT, GPUJob, read, write, append, sha, stamp


def prepare():
    out = EXP/'loss-diagnostic-protocol.json'
    if out.exists():
        raise RuntimeError('Diagnostic sample already frozen')
    source = ROOT/'data/training/functional-v4-20261105/cases.json'
    cases = read(source)
    picked = []
    for domain in ('portrait', 'landscape'):
        for action in ('restore-exposure', 'restore-contrast', 'restore-saturation'):
            pool = [c for c in cases if c['domain'] == domain and c['action'] == action+'-mild']
            pool.sort(key=lambda c: hashlib.sha256(('numeric-loss-diagnosis-v1'+c['id']).encode()).hexdigest())
            for c in pool[:3]:
                picked.extend([c['id'], c['id'][:-4]+'strong'])
    assert len(picked) == 36 and len(set(picked)) == 36
    write(out, dict(frozen_at=stamp(), case_ids=picked, source=str(source.relative_to(ROOT)),
                    source_sha256=sha(source), parent=str(PARENT.relative_to(ROOT)),
                    parent_sha256=sha(PARENT/'adapter_model.safetensors'),
                    objective_sha256=sha(ROOT/'scripts/functional_v5/objective.py'),
                    note='72 teacher-forced training rows, balanced domain/family/severity/language. Loss/token accuracy diagnose supervision, never functional completion. No new weights or acceptance access.'))


def run(gpu):
    protocol = read(EXP/'loss-diagnostic-protocol.json')
    assert sha(ROOT/protocol['source']) == protocol['source_sha256']
    assert sha(PARENT/'adapter_model.safetensors') == protocol['parent_sha256']
    assert sha(ROOT/'scripts/functional_v5/objective.py') == protocol['objective_sha256']
    dest = EXP/'loss-diagnostic.jsonl'
    if dest.exists():
        raise RuntimeError('Keep the existing diagnostic observations')
    cases = {c['id']: c for c in read(ROOT/protocol['source'])}
    with GPUJob(gpu, 'v5-numerical-loss-diagnosis', max_seconds=1800) as budget:
        import torch
        from functional_v1.runtime import processor, load_model
        from .objective import build_numeric_batch, token_losses
        torch.set_num_threads(4)
        p, model = processor(), load_model(PARENT)
        records = []
        for cid in protocol['case_ids']:
            c = cases[cid]
            for language in ('zh', 'en'):
                budget.tick()
                batch = build_numeric_batch(p, c, language, c['parameters'])
                mask = batch.pop('numeric_mask').to('cuda')
                moved = {k: v.to('cuda') for k, v in batch.items()}
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(**moved)
                losses, active, scores, targets = token_losses(output.logits, moved['labels'])
                nums = mask[:, 1:][active]
                assert nums.any() and (~nums).any()
                exact = scores.argmax(-1).eq(targets)
                record = dict(case_id=cid, language=language, domain=c['domain'], action=c['action'],
                              numeric_tokens=int(nums.sum()), other_tokens=int((~nums).sum()),
                              numeric_loss_sum=float(losses[nums].sum()), other_loss_sum=float(losses[~nums].sum()),
                              numeric_correct=int(exact[nums].sum()), other_correct=int(exact[~nums].sum()),
                              builtin_loss=float(output.loss), independent_loss=float(losses.mean()))
                assert abs(record['builtin_loss']-record['independent_loss']) < 1e-4
                append(dest, record)
                records.append(record)
        result = dict(status='completed_diagnostic_not_functional_evaluation', at=stamp(), records=len(records),
                      cases_sha256=protocol['source_sha256'], parent_sha256=protocol['parent_sha256'],
                      protocol_sha256=sha(EXP/'loss-diagnostic-protocol.json'))
        for kind in ('numeric', 'other'):
            count = sum(r[kind+'_tokens'] for r in records)
            result[kind] = dict(tokens=count, mean_loss=sum(r[kind+'_loss_sum'] for r in records)/count,
                                teacher_forced_token_accuracy=sum(r[kind+'_correct'] for r in records)/count)
        write(EXP/'loss-diagnostic-summary.json', result)
        print(result)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['prepare', 'run'])
    ap.add_argument('--gpu', choices=['0', '2', '3'])
    args = ap.parse_args()
    prepare() if args.command == 'prepare' else run(args.gpu)

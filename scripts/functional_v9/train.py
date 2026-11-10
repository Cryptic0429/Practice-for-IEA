"""Matched fresh LoRA training from audited original and author full weights."""
import argparse
import hashlib
import os
import shutil
import traceback
from pathlib import Path

from functional_v1.common import ROOT, read, write, append, sha, stamp, GPUJob, BudgetExceeded
from .runtime import EXP, DATA, verify_origin
from functional_v7.outcomes import EDITOR_VERSION, SCHEMA_VERSION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--route', choices=['base', 'author'], required=True)
    parser.add_argument('--gpu', choices=['0', '2', '3'], required=True)
    parser.add_argument('--seed', type=int, default=20261115)
    args = parser.parse_args()
    protocol = read(EXP / 'training-protocol.json')
    settings = protocol['training']
    assert protocol['status'] == 'actual_data_runtime_optimizer_and_functional_gates_frozen_before_training'
    assert args.seed in (settings['seed'], settings['replicate_seed'])
    if args.seed == settings['replicate_seed']:
        selection = read(EXP / 'development-selection.json')
        assert selection['qualified'] and selection['route'] == args.route
    for name, digest in protocol['code_hashes'].items():
        assert sha(ROOT / name) == digest
    for name, key in [('train-accepted.json', 'dataset_sha256'), ('cases.json', 'cases_sha256'), ('admission.json', 'admission_sha256')]:
        assert sha(DATA / name) == protocol[key]
    assert sha(ROOT / protocol['input_audit_path']) == protocol['input_audit_sha256']
    assert sha(ROOT / protocol['target_replay_path']) == protocol['target_replay_sha256']
    assert sha(EXP / 'evaluation-protocol.json') == protocol['evaluation_protocol_sha256']
    assert sha(EXP / 'origin-audit.json') == protocol['origin_audit_sha256']
    verify_origin(args.route)
    rows = read(DATA / 'train-accepted.json')
    cases = {c['id']: c for c in read(DATA / 'cases.json')}
    assert len(rows) == settings['rows'] and all(r['decision'] == cases[r['case_id']]['decision'] for r in rows)
    out = EXP / 'runs' / f'{args.route}-seed-{args.seed}'
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(Path(__file__).parent, out / 'source', ignore=shutil.ignore_patterns('__pycache__'))
    write(out / 'protocol-snapshot.json', protocol)
    status = dict(status='starting', pid=os.getpid(), route=args.route, gpu=args.gpu, started_at=stamp())
    write(out / 'status.json', status)
    with GPUJob(args.gpu, 'v9-train-' + args.route, max_seconds=5400) as budget:
        with budget.ledger() as value:
            next(j for j in value['jobs'] if j['id'] == budget.id).update(editor=EDITOR_VERSION, schema=SCHEMA_VERSION, runtime='functional_v9.runtime')
        import torch
        from transformers import Trainer, TrainingArguments, TrainerCallback, set_seed
        from .runtime import processor, load_model, build_batch
        torch.set_num_threads(4)
        set_seed(args.seed)
        p = processor()
        prepared = []
        tokens = []
        for row in rows:
            c = cases[row['case_id']]
            assert sha(ROOT / c['image']) == c['input_sha256']
            batch = build_batch(p, c, row['language'], row['decision'])
            tokens.append(dict(id=row['id'], action=row['decision']['action'], input_tokens=int(batch['input_ids'].numel()),
                               assistant_tokens=int(batch['labels'].ne(-100).sum())))
            prepared.append(batch)
        write(out / 'token-audit.json', tokens)
        budget.tick()
        try:
            model = load_model(args.route, train=True, seed=args.seed)
            write(out / 'initialization.json', model.functional_initialization)
            initial = {name: w.detach().cpu().clone() for name, w in model.named_parameters() if w.requires_grad}
            expected_count = 112
            write(out / 'initial-adapter-fingerprints.json', {name: hashlib.sha256(value.numpy().tobytes()).hexdigest() for name, value in initial.items()})
            assert len(initial) == expected_count
            config = TrainingArguments(
                output_dir=str(out), num_train_epochs=settings['epochs'], per_device_train_batch_size=1,
                gradient_accumulation_steps=4, learning_rate=settings['learning_rate'], warmup_ratio=.05,
                lr_scheduler_type='cosine', optim='adamw_torch', weight_decay=0., bf16=True,
                gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
                max_grad_norm=1., save_strategy='epoch', logging_steps=1, report_to='none',
                remove_unused_columns=False, seed=args.seed, data_seed=args.seed, dataloader_num_workers=0, disable_tqdm=True)
            write(out / 'training-arguments.json', config.to_dict())

            class Audit(TrainerCallback):
                def on_pre_optimizer_step(self, arguments, state, control, **kwargs):
                    active = {name: w for name, w in model.named_parameters() if w.requires_grad}
                    assert set(active) == set(initial) and len(active) == expected_count
                    assert all(w.dtype == torch.float32 and w.grad is not None and torch.isfinite(w.grad).all() for w in active.values())
                    if state.global_step == 0:
                        optimizer_ids = {id(w) for group in kwargs['optimizer'].param_groups for w in group['params']}
                        assert optimizer_ids == {id(w) for w in active.values()}
                        gradients = {name: dict(norm=float(w.grad.norm()), nonzero=int(torch.count_nonzero(w.grad))) for name, w in active.items()}
                        write(out / 'first-gradient-check.json', dict(at=stamp(), finite=True, tensors=expected_count,
                                                                      optimizer_contains_only_adapter=True, gradients=gradients))

                def on_step_end(self, arguments, state, control, **kwargs):
                    status.update(status='running', step=state.global_step, epoch=state.epoch, at=stamp())
                    write(out / 'status.json', status)
                    try:
                        budget.tick()
                    except BudgetExceeded:
                        control.should_training_stop = True
                        control.should_save = True
                        status['budget_stop_requested'] = True

                def on_log(self, arguments, state, control, logs=None, **kwargs):
                    append(out / 'metrics.jsonl', dict(step=state.global_step, **(logs or {})))

            trainer = Trainer(model=model, args=config, train_dataset=prepared, data_collator=lambda batch: batch[0], callbacks=[Audit()])
            set_seed(args.seed)  # Adapter construction consumes different RNG amounts; reset before the matched data order.
            result = trainer.train()
            trainer.save_model(str(out / 'final'))
            trainer.save_state()
            changes = {name: not torch.equal(initial[name], w.detach().cpu()) for name, w in model.named_parameters() if name in initial}
            assert any(v for name, v in changes.items() if 'visual' not in name)
            assert sum(changes.values()) == 112
            assert all(torch.isfinite(w).all() for w in model.parameters() if w.requires_grad)
            assert status.get('budget_stop_requested') or trainer.state.global_step == settings['steps']
            status.update(status='budget_stopped' if status.get('budget_stop_requested') else 'completed',
                          step=trainer.state.global_step, finished_at=stamp(), metrics=result.metrics,
                          changed_language_tensors=sum(v for name, v in changes.items() if 'visual' not in name),
                          changed_visual_merger_tensors=sum(v for name, v in changes.items() if 'visual' in name),
                          adapter_sha256=sha(out / 'final/adapter_model.safetensors'))
            write(out / 'status.json', status)
        except BaseException as error:
            status.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc(), at=stamp())
            write(out / 'status.json', status)
            raise


if __name__ == '__main__':
    main()

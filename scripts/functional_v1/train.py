"""Matched pilots; requires every target to pass the recorded primary visual review."""
from __future__ import annotations
import argparse
import collections
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from .common import ROOT, DATA, EXP, PARENT, GPUJob, read, write, append, sha, stamp, EDITOR_VERSION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', choices=['base', 'current'], required=True)
    ap.add_argument('--gpu', choices=['0','2','3'], required=True)
    ap.add_argument('--seed', type=int, default=20261104)
    args = ap.parse_args()
    acceptance = read(DATA/'label-acceptance.json')
    assert acceptance['status'] == 'accepted_for_pilot_training'
    assert acceptance['dataset_sha256'] == sha(DATA/'train-accepted.json')
    rows = read(DATA/'train-accepted.json')
    assert len(rows) == 440 and len({r['case_id'] for r in rows}) == 220
    cases = {c['id']: c for c in read(DATA/'train-cases.json')}
    out = EXP/'runs'/f'{args.start}-seed-{args.seed}'
    out.mkdir(parents=True, exist_ok=False)
    source_dir = out/'source'
    shutil.copytree(Path(__file__).parent, source_dir, ignore=shutil.ignore_patterns('__pycache__'))
    settings = {'start': args.start, 'seed': args.seed, 'epochs': 4,
                'learning_rate': 1e-4, 'effective_batch': 4, 'rank': 8, 'alpha': 16,
                'tools_editor': EDITOR_VERSION, 'rows': 440, 'semantic_cases': 220,
                'dataset_sha256': sha(DATA/'train-accepted.json'),
                'parent_adapter_sha256': sha(PARENT/'adapter_model.safetensors') if args.start == 'current' else None,
                'selection': 'Development instruction completion and image quality; no L1 checkpoint selection.'}
    write(out/'settings.json', settings)
    status = {'status': 'starting', 'pid': os.getpid(), 'gpu': args.gpu, 'started_at': stamp()}
    write(out/'status.json', status)
    with GPUJob(args.gpu, 'train-'+out.name, max_seconds=7200) as budget:
        import torch
        from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
        from .runtime import load_model, processor, build_batch
        torch.set_num_threads(4)
        set_seed(args.seed)
        p = processor()
        prepared, audit = [], []
        for row in rows:
            batch = build_batch(p, cases[row['case_id']], row['language'], row['parameters'])
            audit.append({'id': row['id'], 'input_tokens': int(batch['input_ids'].numel()),
                          'supervised_tokens': int((batch['labels'] != -100).sum())})
            prepared.append(batch)
        write(out/'token-audit.json', audit)
        budget.tick()
        model = load_model(PARENT if args.start == 'current' else None, train=True)
        initial = {k: v.detach().cpu().clone() for k,v in model.named_parameters() if v.requires_grad}
        write(out/'trainable-parameters.json', [{'name': n, 'dtype': str(v.dtype), 'shape': list(v.shape)}
                                               for n,v in model.named_parameters() if v.requires_grad])
        config = TrainingArguments(output_dir=str(out), num_train_epochs=4,
            per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=1e-4,
            warmup_ratio=.05, lr_scheduler_type='cosine', optim='adamw_torch', weight_decay=0.,
            bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
            max_grad_norm=1., save_strategy='epoch', logging_steps=1, report_to='none',
            remove_unused_columns=False, seed=args.seed, data_seed=args.seed,
            dataloader_num_workers=0, disable_tqdm=True)
        write(out/'training-arguments.json', config.to_dict())
        class Audit(TrainerCallback):
            def on_pre_optimizer_step(self, args, state, control, **kwargs):
                active = [v for v in model.parameters() if v.requires_grad]
                assert len(active) == 112 and all(v.dtype == torch.float32 for v in active)
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in active)

            def on_step_end(self, args, state, control, **kwargs):
                status.update(status='running', step=state.global_step, epoch=state.epoch, updated_at=stamp())
                write(out/'status.json', status)
                try:
                    budget.tick()
                except Exception:
                    control.should_training_stop = True
                    control.should_save = True
                    status['budget_stop_requested'] = True

            def on_log(self, args, state, control, logs=None, **kwargs):
                append(out/'metrics.jsonl', {'step': state.global_step, **(logs or {})})
        try:
            trainer = Trainer(model=model, args=config, train_dataset=prepared,
                              data_collator=lambda b: b[0], callbacks=[Audit()])
            result = trainer.train()
            trainer.save_model(str(out/'final'))
            trainer.save_state()
            changed = any(not torch.equal(initial[n], v.detach().cpu()) for n,v in model.named_parameters() if n in initial)
            assert changed
            completed = trainer.state.global_step == 440 and not status.get('budget_stop_requested')
            status.update(status='completed' if completed else 'budget_stopped',
                          step=trainer.state.global_step, metrics=result.metrics, weights_changed=changed,
                          adapter_sha256=sha(out/'final/adapter_model.safetensors'), finished_at=stamp())
            write(out/'status.json', status)
        except BaseException as error:
            status.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
            write(out/'status.json', status)
            raise


if __name__ == '__main__':
    main()

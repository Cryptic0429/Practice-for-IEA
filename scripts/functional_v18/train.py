"""Matched full-positive SFT versus visually admitted counterfactual ranking."""
import argparse
import hashlib
import os
import shutil
import signal
import traceback
from pathlib import Path
from collections import Counter

from functional_v1.common import ROOT, read, write, append, sha, stamp, GPUJob, BudgetExceeded
from .runtime import EXP, DATA, verify_origin, ROUTES, expected_count
from functional_v15.outcomes import EDITOR_VERSION, SCHEMA_VERSION
from .prepare_training import streaming_sha
from .objective import sequence_logp, ranking_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--route', choices=ROUTES, required=True)
    parser.add_argument('--gpu', choices=['0', '2', '3'], required=True)
    parser.add_argument('--seed', type=int, default=20261129)
    args = parser.parse_args()
    protocol = read(EXP / 'training-protocol.json')
    settings = protocol['training']
    assert protocol['status'] == 'visual_preferences_matched_optimization_and_functional_gates_frozen_before_training'
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
    verify_origin()
    assert streaming_sha(ROOT/protocol['base_cache_path'])==protocol['base_cache_sha256']
    assert streaming_sha(ROOT/protocol['negative_cache_path'])==protocol['negative_cache_sha256']
    assert sha(DATA/'preferences.json')==protocol['preferences_sha256']
    assert sha(DATA/'training-pool.json')==protocol['training_pool_sha256']
    rows = read(DATA / 'train-accepted.json')
    cases = {c['id']: c for c in read(DATA / 'cases.json')}
    assert len(rows) == 1850 and all(r['decision'] == cases[r['case_id']]['decision'] for r in rows)
    import torch
    torch.set_num_threads(4)
    base_cache=torch.load(ROOT/protocol['base_cache_path'],map_location='cpu',weights_only=True)
    negative_cache=torch.load(ROOT/protocol['negative_cache_path'],map_location='cpu',weights_only=True)
    assert len(base_cache)==1850 and len(negative_cache)==38
    prepared=[]
    for row in read(DATA/'training-pool.json'):
        item=dict(id=row['id'],kind=row['kind'],chosen=base_cache[row['base_row_index']])
        if row['kind']=='preference':item['rejected']=negative_cache[row['preference_id']]
        prepared.append(item)
    assert len(prepared)==settings['rows']==2002
    out = EXP / 'runs' / f'{args.route}-seed-{args.seed}'
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(Path(__file__).parent, out / 'source', ignore=shutil.ignore_patterns('__pycache__'))
    write(out / 'protocol-snapshot.json', protocol)
    status = dict(status='starting', pid=os.getpid(), route=args.route, gpu=args.gpu, started_at=stamp())
    write(out / 'status.json', status)
    stop_signal = {'requested': False, 'signal': None}
    def request_stop(signum, frame):
        stop_signal.update(requested=True, signal=signum)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    with GPUJob(args.gpu, 'v18-train-' + args.route, max_seconds=3600) as budget:
        with budget.ledger() as value:
            next(j for j in value['jobs'] if j['id'] == budget.id).update(editor=EDITOR_VERSION, schema=SCHEMA_VERSION, runtime='functional_v18.runtime')
        from transformers import Trainer, TrainingArguments, TrainerCallback, set_seed
        from .runtime import processor, load_model, build_batch
        torch.set_num_threads(4)
        set_seed(args.seed)
        write(out/'token-audit.json',read(EXP/'training-input-cache-audit.json'))
        budget.tick()
        try:
            model = load_model(args.route, train=True, seed=args.seed)
            write(out / 'initialization.json', model.functional_initialization)
            initial = {name: w.detach().cpu().clone() for name, w in model.named_parameters() if w.requires_grad}
            tensor_count = expected_count(args.route)
            write(out / 'initial-adapter-fingerprints.json', {name: hashlib.sha256(value.numpy().tobytes()).hexdigest() for name, value in initial.items()})
            assert len(initial) == tensor_count
            calls=Counter()

            class CorrectionTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                    chosen_batch=inputs['chosen']
                    output=model(**{k:v for k,v in chosen_batch.items() if k!='labels'})
                    chosen,count=sequence_logp(output.logits,chosen_batch['labels'])
                    loss=-chosen/count
                    calls[inputs['kind']]+=1
                    if args.route=='counterfactual_rank' and inputs['kind']=='preference':
                        rejected_batch=inputs['rejected']
                        rejected_output=model(**{k:v for k,v in rejected_batch.items() if k!='labels'})
                        rejected,rejected_count=sequence_logp(rejected_output.logits,rejected_batch['labels'])
                        loss=loss+ranking_loss(chosen,rejected,count,rejected_count,
                            settings['ranking_weight'],settings['ranking_margin'],settings['ranking_score_scale'])
                        calls['actual_rejected_forward']+=1
                    assert torch.isfinite(loss)
                    return (loss,output) if return_outputs else loss

            # Independently compare assistant-only CE with the model's built-in
            # causal loss on an actual admitted preference, before any update.
            first=next(r for r in prepared if r['kind']=='preference')
            sample={k:v.to('cuda') for k,v in first['chosen'].items()}
            model.eval()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                checked=model(**sample)
                logp,count=sequence_logp(checked.logits,sample['labels'])
            assert torch.allclose(-logp/count,checked.loss.float(),atol=1e-4,rtol=1e-4)
            write(out/'real-mask-preflight.json',dict(at=stamp(),passed=True,assistant_tokens=count,
                independent_assistant_CE=float(-logp/count),builtin_causal_CE=float(checked.loss),optimizer_steps=0))
            del checked,sample,logp
            # Fresh TRAIN-only initial function check: prepared numerics must not
            # silently change these predeclared greedy parent decisions.
            import json
            from .runtime import predict
            frozen_predictions=ROOT/protocol['parent_diagnostic_predictions_path']
            assert sha(frozen_predictions)==protocol['parent_diagnostic_predictions_sha256']
            parent_predictions={r['case_id']:r for r in [json.loads(s) for s in frozen_predictions.read_text().splitlines()]}
            proc=processor();parity=[]
            for cid in protocol['initial_parity_TRAIN_case_ids']:
                response,decision=predict(model,proc,cases[cid],'en')
                parity.append(dict(case_id=cid,response=response,decision=decision,
                    same_response_as_reloaded_parent=response==parent_predictions[cid]['response']))
                budget.tick()
            write(out/'initial-function-parity.json',dict(at=stamp(),records=parity,optimizer_steps=0,
                scope='3 fixed TRAIN prefixes only, not universal numeric equivalence or editing quality'))
            assert all(r['same_response_as_reloaded_parent'] for r in parity), 'Stop before optimizer: initial greedy function differs'
            model.train()
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
                    assert set(active) == set(initial) and len(active) == tensor_count
                    assert all(w.dtype == torch.float32 and w.grad is not None and torch.isfinite(w.grad).all() for w in active.values())
                    if state.global_step == 0:
                        optimizer_ids = {id(w) for group in kwargs['optimizer'].param_groups for w in group['params']}
                        assert optimizer_ids == {id(w) for w in active.values()}
                        gradients = {name: dict(norm=float(w.grad.norm()), nonzero=int(torch.count_nonzero(w.grad))) for name, w in active.items()}
                        write(out / 'first-gradient-check.json', dict(at=stamp(), finite=True, tensors=tensor_count,
                                                                      optimizer_contains_only_adapter=True, gradients=gradients))

                def on_step_end(self, arguments, state, control, **kwargs):
                    status.update(status='running', step=state.global_step, epoch=state.epoch, at=stamp())
                    write(out / 'status.json', status)
                    if stop_signal['requested']:
                        control.should_training_stop = True
                        control.should_save = True
                        status.update(budget_stop_requested=True, stop_signal=stop_signal['signal'])
                    try:
                        budget.tick()
                    except BudgetExceeded:
                        control.should_training_stop = True
                        control.should_save = True
                        status['budget_stop_requested'] = True

                def on_log(self, arguments, state, control, logs=None, **kwargs):
                    append(out / 'metrics.jsonl', dict(step=state.global_step, **(logs or {})))

            trainer = CorrectionTrainer(model=model, args=config, train_dataset=prepared, data_collator=lambda batch: batch[0], callbacks=[Audit()])
            trainer.model_accepts_loss_kwargs=False
            set_seed(args.seed)  # Adapter construction consumes different RNG amounts; reset before the matched data order.
            result = trainer.train()
            trainer.save_model(str(out / 'final'))
            trainer.save_state()
            if status.get('budget_stop_requested'):
                trainer._save_checkpoint(model, trial=None)  # Include optimizer/scheduler/RNG for interrupted step.
                write(out/'resume-checkpoint.json', dict(step=trainer.state.global_step, signal=stop_signal['signal'], optimizer_state_saved=True))
            changes = {name: not torch.equal(initial[name], w.detach().cpu()) for name, w in model.named_parameters() if name in initial}
            assert any(v for name, v in changes.items() if 'visual' not in name)
            assert sum(changes.values()) == tensor_count
            assert all(torch.isfinite(w).all() for w in model.parameters() if w.requires_grad)
            assert status.get('budget_stop_requested') or trainer.state.global_step == settings['steps']
            status.update(status='budget_stopped' if status.get('budget_stop_requested') else 'completed',
                          step=trainer.state.global_step, finished_at=stamp(), metrics=result.metrics,
                          changed_language_tensors=sum(v for name, v in changes.items() if 'visual' not in name),
                          changed_visual_merger_tensors=sum(v for name, v in changes.items() if 'visual' in name),
                          adapter_sha256=sha(out / 'final/adapter_model.safetensors'))
            status['actual_objective_calls']=dict(calls)
            if not status.get('budget_stop_requested'):
                assert calls['anchor']==3700 and calls['preference']==304
                assert calls['actual_rejected_forward']==(304 if args.route=='counterfactual_rank' else 0)
            write(out / 'status.json', status)
        except BaseException as error:
            status.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc(), at=stamp())
            write(out / 'status.json', status)
            raise


if __name__ == '__main__':
    main()

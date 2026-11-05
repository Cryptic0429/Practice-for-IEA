"""Measured TRL GRPO with frozen-A reference and acceptance-gated formal mode."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import shutil
import time
import traceback

import torch
from datasets import Dataset
from peft import set_peft_model_state_dict
from transformers import TrainerCallback, set_seed
from trl import GRPOConfig, GRPOTrainer
from stage2_runtime import (ROOT, load_model, processor, prompt, image_input, encoded,
                           adapter_state, tensor_digest, FrozenReference, chunk_generation, assert_trainable)
from stage2_reward import ImageReward


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write(p, data):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(p)


class AuditedTrainer(GRPOTrainer):
    def _generate_and_score_completions(self, inputs):
        # Quantized training keeps selected frozen layers in FP32; generation is
        # outside Trainer.compute_loss's autocast context, so supply the same BF16 context.
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = super()._generate_and_score_completions(inputs)
        # TRL 0.23 unwrap_model_for_generation re-enables checkpointing without its
        # kwargs, resetting to reentrant=True. Frozen embeddings need nonreentrant.
        self.accelerator.unwrap_model(self.model).gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant':False})
        return result

    def _get_per_token_logps_and_entropies(self, *args, **kwargs):
        kwargs['batch_size'] = 1
        return super()._get_per_token_logps_and_entropies(*args, **kwargs)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--run-name', required=True)
    cli.add_argument('--generation-concurrency', type=int, choices=[1,2,4], default=1,
                     help='Preflight resource setting; formal uses the frozen gate')
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--formal', action='store_true')
    args = cli.parse_args()
    target_steps = 8 if args.preflight else 128
    expected_rows = 32 if args.preflight else 512
    max_seconds = 3600 if args.preflight else 12*3600
    gate = None
    if args.formal:
        from stage2_budget import ledger
        gate = json.loads((ROOT/'configs/stage2-formal-gate.json').read_text())
        assert gate['passed']
        import hashlib
        for path,digest in gate['evidence'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
        resources = ledger()
        assert resources['budget_safe_to_launch']
        assert resources['remaining_gpu_hours'] >= gate['projected_formal_reserved_gpu_hours'] + gate['reserve_other_gpu_hours']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '2'
    assert int(subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.used',
                                      '--format=csv,noheader,nounits'], text=True)) < 512
    out = ROOT/'outputs/stage2-A-edit-grpo-v1'/args.run_name
    out.mkdir(parents=True, exist_ok=False)
    lock = (out/'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    status = dict(status='loading', pid=os.getpid(), gpu=2, started_at=stamp(), updated_at=stamp(), step=0)
    write(out/'status.json', status)
    (out/'source').mkdir()
    for name in ['train_stage2_grpo.py','stage2_runtime.py','stage2_reward.py','evaluation_core.py',
                 'evaluate_stage2.py','stage2_budget.py']:
        shutil.copy2(ROOT/'scripts'/name,out/'source'/name)
    if gate:
        write(out/'formal-gate.json',gate)
    gpu = subprocess.Popen(['nvidia-smi','-i','2','--query-gpu=timestamp,memory.used,temperature.gpu,utilization.gpu',
                            '--format=csv,noheader,nounits','--loop=2'], stdout=(out/'gpu.csv').open('w'))
    try:
        data_dir = ROOT/'data/stage2/stage2-A-edit-grpo-v1'
        manifest = json.loads((data_dir/'manifest.json').read_text())
        import hashlib
        for item in manifest['files'].values():
            assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
        source_name = 'preflight.json' if args.preflight else 'train.json'
        rows = json.loads((data_dir/source_name).read_text())
        assert len(rows) == expected_rows
        protocol = json.loads((ROOT/'configs/evaluation-protocol-v1.json').read_text())
        for path, expected in protocol['code_hashes'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
        rewards = ImageReward(ROOT, protocol, rows)
        set_seed(20261101)
        torch.set_num_threads(4)
        model = load_model()
        initial = adapter_state(model)
        reference = adapter_state(model,'reference')
        assert tensor_digest(initial) == tensor_digest(reference)
        processing = processor()
        # Match original Stage 1 CPU input audit before allowing any update.
        parity = ROOT/'logs/stage2-A-edit-grpo-v1/input-parity.json'
        assert json.loads(parity.read_text())['all_equal'], 'Input parity must be accepted first'
        records, groups = [], []
        dataset_rows = []
        for r in rows:
            user = next(m['content'] for m in r['conversations'] if m['role']=='user')
            assert user.startswith('<image>') and user.count('<image>') == 1
            messages = [dict(role=m['role'],content=m['content'].replace('<image>',''))
                        for m in r['conversations'] if m['role'] in ('system','user')]
            dataset_rows.append(dict(id=r['id'],prompt=messages,image_path=r['images'][0]))
        dataset = Dataset.from_list(dataset_rows)
        def transform(batch):
            batch['image'] = [image_input(p) for p in batch.pop('image_path')]
            return batch
        dataset.set_transform(transform)
        config = GRPOConfig(output_dir=str(out), per_device_train_batch_size=1,
            gradient_accumulation_steps=16, generation_batch_size=16, num_generations=4,
            max_steps=target_steps, max_completion_length=512, max_prompt_length=None,
            learning_rate=1e-6, warmup_ratio=.05, lr_scheduler_type='cosine', optim='adamw_torch', weight_decay=0.,
            beta=.01, temperature=.8, top_p=.9, top_k=0, loss_type='grpo', scale_rewards='group',
            epsilon=.2, num_iterations=1, bf16=True, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False}, max_grad_norm=1.,
            save_strategy='steps', save_steps=8 if args.preflight else 64, logging_steps=1, report_to='none',
            remove_unused_columns=False, dataloader_num_workers=0, seed=20261101, data_seed=20261101,
            use_vllm=False, disable_tqdm=True)
        write(out/'run-config.json',config.to_dict())
        if gate:
            frozen=gate['formal_settings']
            for field in ['learning_rate','warmup_ratio','temperature','top_p','top_k','beta','loss_type',
                          'scale_rewards','epsilon','num_iterations','max_grad_norm','seed']:
                assert getattr(config,field)==frozen[field], field
            assert config.get_warmup_steps(target_steps)==7
        def phase(name, **details):
            status.update(status=name,updated_at=stamp(),**details)
            write(out/'status.json',status)
        def reward_func(prompts, completions, id, completion_ids, trainer_state, **kwargs):
            assert len(id) == 16 and len(set(id)) == 4
            assert all(len(set(id[i:i+4])) == 1 for i in range(0,16,4))
            batch = []
            for sample, completion, tokens in zip(id,completions,completion_ids):
                text = completion[0]['content'] if isinstance(completion,list) else completion
                record = rewards.score(sample,text)
                record.update(step=trainer_state.global_step+1, candidate=len(batch)%4,
                              completion_ids=tokens, output_tokens=len(tokens),length_limit_reached=len(tokens)>=512,
                              truncated=len(tokens)>=512 and processing.tokenizer.eos_token_id not in tokens)
                batch.append(record)
            for i in range(0,16,4):
                group = batch[i:i+4]
                groups.append(dict(id=id[i], step=trainer_state.global_step+1,
                              distinct_outputs=len({x['response'] for x in group}),
                              distinct_rewards=len({x['reward'] for x in group})))
            with (out/'candidates.jsonl').open('a') as f:
                for record in batch:
                    f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            records.extend(batch)
            status.update(status='optimizing', updated_at=stamp(), candidates=len(records))
            write(out/'status.json',status)
            return [x['reward'] for x in batch]
        trainer = AuditedTrainer(model=model,args=config,train_dataset=dataset,
                                 processing_class=processing,reward_funcs=reward_func)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        # Preserve exact FP32 A adapters while TRL prepares the mixed-precision base.
        for name,param in model.named_parameters():
            if 'lora_' in name:
                param.data = param.data.float()
        set_peft_model_state_dict(model,initial,adapter_name='default')
        set_peft_model_state_dict(model,reference,adapter_name='reference')
        assert tensor_digest(adapter_state(model)) == tensor_digest(initial)
        trainer.ref_model = FrozenReference(model)
        assert_trainable(model)
        probe = {k:v.to('cuda') for k,v in encoded(processing,rows[0]).items()}
        model.eval()
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
            policy_logits = model(**probe,use_cache=False).logits[:,-1].float()
            ref_logits = trainer.ref_model(**probe,use_cache=False).logits[:,-1].float()
        assert torch.equal(policy_logits,ref_logits), 'Initial A probabilities mismatch'
        assert torch.isfinite(policy_logits).all()
        write(out/'initial-policy-check.json',dict(exact_logits_equal=True,
              adapter_tensors=len(initial), initial_adapter_digest=tensor_digest(initial),
              reference_adapter_digest=tensor_digest(reference), trainable=assert_trainable(model)))
        del policy_logits, ref_logits, probe
        # Check a real training-input gradient before paying for candidate generation.
        model.train()
        probe = {k:v.to('cuda') for k,v in encoded(processing,rows[0]).items()}
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(**probe,use_cache=False).logits[:,-1].float()
            gradient_probe=logits.logsumexp(-1).mean()
        assert gradient_probe.requires_grad
        gradient_probe.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        model.zero_grad(set_to_none=True)
        write(out/'gradient-probe.json',dict(passed=True,optimizer_updates=0,
             finite_trainable_gradients=112,source_id=rows[0]['id']))
        del probe,logits,gradient_probe
        generation_concurrency=gate['formal_settings']['generation_concurrency'] if gate else args.generation_concurrency
        chunk_generation(model,concurrency=generation_concurrency)
        validations = []
        class Audit(TrainerCallback):
            def on_step_begin(self,training_args,state,control,**kwargs):
                phase('generating',next_step=state.global_step+1)
            def on_pre_optimizer_step(self,args,state,control,**kwargs):
                assert_trainable(model)
                gradients=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                assert len(gradients)==112 and all(torch.isfinite(g).all() for g in gradients)
                assert tensor_digest(adapter_state(model,'reference')) == tensor_digest(reference)
            def on_step_end(self,args,state,control,**kwargs):
                status.update(status='running',step=state.global_step,updated_at=stamp(),
                              elapsed_seconds=time.monotonic()-started,
                              peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                              peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                write(out/'status.json',status)
                elapsed=time.monotonic()-started
                exhausted=elapsed > max_seconds
                if gate:
                    resource=ledger()
                    # Preserve at least one recent step's time as a stop margin.
                    step_estimate=max(300,elapsed/max(1,state.global_step))
                    exhausted |= elapsed+step_estimate >= max_seconds
                    validation_remaining=gate['estimated_validation_wall_hours']*(2-len(validations))/2
                    test_and_control=gate['reserve_other_gpu_hours']-gate['estimated_validation_wall_hours']
                    # Reserve both allocated GPUs during future validations, plus
                    # the required SFT control and all final comparisons.
                    reserve_seconds=(2*validation_remaining+test_and_control)*3600
                    exhausted |= resource['remaining_gpu_hours']*3600 < step_estimate+reserve_seconds
                if exhausted:
                    control.should_training_stop=True
                    control.should_save=True
            def on_save(self,training_args,state,control,**kwargs):
                checkpoint=out/f'checkpoint-{state.global_step}'
                required=['adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json']
                assert all((checkpoint/name).exists() for name in required)
                if not gate or state.global_step not in [64,128]:
                    return
                phase('validating',step=state.global_step)
                # GPU0 may belong to another job. Wait without touching it, bounded by own budget.
                while int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used',
                                                   '--format=csv,noheader,nounits'],text=True))>=512:
                    if time.monotonic()-started>max_seconds-3600:
                        raise RuntimeError('No free validation GPU within remaining formal budget; checkpoint retained')
                    phase('waiting_for_validation_gpu',step=state.global_step)
                    time.sleep(10)
                remaining=ledger()['remaining_gpu_hours']*3600
                test_and_control=(gate['reserve_other_gpu_hours']-gate['estimated_validation_wall_hours'])*3600
                allowance=min(3600,(max_seconds-(time.monotonic()-started)),max(0,(remaining-test_and_control)/2))
                assert allowance>300, 'Insufficient budget for full validation; checkpoint retained'
                name=f'GRPO-step{state.global_step}'
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
                command=[str(ROOT/'envs/iea-rl/bin/python'),'-u',str(ROOT/'scripts/evaluate_stage2.py'),
                         '--adapter',str(checkpoint),'--name',name,'--split','validation','--gpu','0',
                         '--max-gpu-seconds',str(allowance)]
                with (out/f'validation-{state.global_step}.log').open('w') as log:
                    child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                    phase('validating',validation_pid=child.pid,step=state.global_step)
                    assert child.wait()==0, 'Validation failed; checkpoint and validation evidence retained'
                directory=ROOT/'outputs/stage2-A-edit-grpo-v1/evaluation'/f'validation-{name}'
                summary=json.loads((directory/'summary.json').read_text())
                assert summary['rows']==200
                validations.append(dict(step=state.global_step,checkpoint=str(checkpoint),
                                        mean_l1=summary['mean_l1'],summary_path=str(directory/'summary.json')))
                write(out/'validation-selection.json',dict(validations=validations,
                      best=min(validations,key=lambda item:(item['mean_l1'],item['step']))))
                phase('running',step=state.global_step)
            def on_log(self,args,state,control,logs=None,**kwargs):
                with (out/'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(dict(step=state.global_step,at=stamp(),**(logs or {})),allow_nan=False)+'\n')
        trainer.add_callback(Audit())
        model.train()
        result = trainer.train()
        trainer.save_model(str(out/'final'))
        trainer.save_state()
        complete=trainer.state.global_step==target_steps
        assert len(records)==trainer.state.global_step*16
        if complete:
            assert Counter(r['id'] for r in records)==Counter({r['id']:4 for r in rows})
        else:
            write(out/'budget-stop.json',dict(actual_step=trainer.state.global_step,target_steps=target_steps,
                  candidates=len(records),resumable_checkpoint=str(out/f'checkpoint-{trainer.state.global_step}')))
        assert tensor_digest(adapter_state(model)) != tensor_digest(initial)
        assert tensor_digest(adapter_state(model,'reference')) == tensor_digest(reference)
        torch.save(dict(input=dict(encoded(processing,rows[0])), sample_id=rows[0]['id']),out/'reload-probe.pt')
        model.eval()
        probe={k:v.to('cuda') for k,v in encoded(processing,rows[0]).items()}
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(**probe,use_cache=False).logits[:,-1].float().cpu()
        torch.save(logits,out/'reload-expected-logits.pt')
        write(out/'training-verification.json',dict(status='awaiting_independent_reload' if complete else 'budget_stopped',
              step=trainer.state.global_step,candidates=len(records),target_steps=target_steps,
              generation_concurrency=generation_concurrency,left_padding_removed_for_generation=True,
              groups=groups, reference_unchanged=True, policy_changed=True,
              training_metrics=result.metrics, elapsed_seconds=time.monotonic()-started,
              gpu_hours=(time.monotonic()-started)/3600,
              final_adapter_digest=tensor_digest(adapter_state(model)),
              peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
              peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        if gate and complete:
            assert {v['step'] for v in validations}=={64,128}
            best=min(validations,key=lambda item:(item['mean_l1'],item['step']))
            selected=out/'selected';selected.mkdir()
            for filename in ['adapter_config.json','adapter_model.safetensors']:
                shutil.copy2(Path(best['checkpoint'])/filename,selected/filename)
            write(out/'selected-model.json',dict(**best,selected_adapter=str(selected),
                  adapter_sha256=hashlib.sha256((selected/'adapter_model.safetensors').read_bytes()).hexdigest()))
        status.update(status='awaiting_independent_reload' if complete else 'budget_stopped',updated_at=stamp(),
                      elapsed_seconds=time.monotonic()-started,gpu_hours=(time.monotonic()-started)/3600)
        write(out/'status.json',status)
    except BaseException as exc:
        status.update(status='failed',updated_at=stamp(),error=str(exc),traceback=traceback.format_exc(),
                      elapsed_seconds=time.monotonic()-started)
        write(out/'status.json',status)
        raise
    finally:
        gpu.terminate()
        gpu.wait(timeout=10)


if __name__ == '__main__':
    main()

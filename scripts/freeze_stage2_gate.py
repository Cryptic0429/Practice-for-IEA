"""Accept real preflight evidence before authorizing the frozen formal workload."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from stage2_budget import ROOT,ledger


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--preflight',required=True);args=cli.parse_args()
    run=ROOT/'outputs/stage2-A-edit-grpo-v1'/args.preflight
    read=lambda p:json.loads(p.read_text())
    training=read(run/'training-verification.json');reload=read(run/'reload-verification.json')
    assert training['step']==8 and training['candidates']==128 and training['reference_unchanged'] and training['policy_changed']
    assert training['left_padding_removed_for_generation'], 'Reject preflight affected by padded generation bug'
    assert reload['formal_gate_passed'] and reload['forward_exact'] and reload['adapter_tensors_exact']
    candidates=read(run/'candidate-audit.json')
    checkpoint=read(run/'checkpoint-audit.json')
    assert candidates['passed'] and candidates['candidates']==128 and candidates['prompts']==32
    assert checkpoint['passed'] and checkpoint['step']==8 and checkpoint['optimizer_parameter_states']==112
    padding_report=ROOT/'outputs/stage2-A-edit-grpo-v1/padding-diagnostic-fixed-all/result.json'
    padding=read(padding_report)
    assert padding['passed'] and padding['prompts']==32 and padding['concurrency']==training['generation_concurrency']
    assert read(ROOT/'logs/stage2-A-edit-grpo-v1/input-parity.json')['all_equal']
    assert read(ROOT/'logs/stage2-A-edit-grpo-v1/reward-parity.json')['all_equal']
    assert read(ROOT/'logs/stage2-A-edit-grpo-v1/batch-input-parity.json')['passed']
    assert read(ROOT/'logs/stage2-A-edit-grpo-v1/input-weight-verification.json')['passed']
    metrics=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
    assert {x['step'] for x in metrics}==set(range(1,9))
    assert all(math.isfinite(v) for r in metrics for v in r.values() if isinstance(v,(float,int)))
    resources=ledger();assert resources['budget_safe_to_launch']
    resource_probe=ROOT/'outputs/stage2-A-edit-grpo-v1/generation-resource-probe/result.json'
    probe=read(resource_probe)
    assert probe['passed'] and probe['training_updates']==0 and probe['policy_unchanged']
    baseline=next(s for s in probe['samples'] if s['concurrency']==training['generation_concurrency'])
    feasible=[s for s in probe['samples'] if s['peak_reserved_gib']<=18 and s['peak_allocated_gib']<=18]
    selected=max(feasible,key=lambda s:s['tokens_per_second'])
    assert selected['concurrency']==training['generation_concurrency'], 'Run acceptance preflight at chosen concurrency'
    measured_speedup=selected['tokens_per_second']/baseline['tokens_per_second']
    conservative_speedup=min(2.,measured_speedup)
    # Approximation includes the loading overhead 16 times, deliberately conservative.
    candidate_seconds=training['elapsed_seconds']/128
    records=[json.loads(line) for line in (run/'candidates.jsonl').read_text().splitlines()]
    mean_tokens=sum(r['output_tokens'] for r in records)/len(records)
    serial=next(s for s in probe['samples'] if s['concurrency']==1)
    serial_seconds=max(mean_tokens/serial['tokens_per_second'],serial['seconds']/serial['completions'])*1.25
    # Only75% of measured runtime is credited with the generation speedup; retain
    #25% unchanged and cap extrapolated speedup at2x because this is one prompt.
    projected_optimization=training['elapsed_seconds']*16*(.25+.75/conservative_speedup)/3600
    # Validation/tests generate serially. Never transfer4-way preflight throughput
    # to these jobs. Use measured1-way token speed and actual preflight length,
    # with25% margin; retain the resource probe's own per-example time as a floor.
    validation_hours=400*serial_seconds/3600
    projected_formal_reserved=projected_optimization+validation_hours
    remaining_for_other=(400+900)*serial_seconds/3600 + .5 + .1
    assert projected_formal_reserved<=12, 'Measure a faster resource-only execution strategy before formal launch'
    assert projected_formal_reserved+remaining_for_other<=resources['remaining_gpu_hours'], 'Insufficient measured budget reserve'
    files=[run/'training-verification.json',run/'reload-verification.json',run/'initial-policy-check.json',
           run/'candidate-audit.json',run/'checkpoint-audit.json',
           padding_report,
           ROOT/'logs/stage2-A-edit-grpo-v1/input-parity.json',ROOT/'logs/stage2-A-edit-grpo-v1/reward-parity.json',
           ROOT/'logs/stage2-A-edit-grpo-v1/input-weight-verification.json',
           ROOT/'logs/stage2-A-edit-grpo-v1/batch-input-parity.json',
           resource_probe,
           ROOT/'data/stage2/stage2-A-edit-grpo-v1/manifest.json',ROOT/'configs/stage2-rl-installed.txt',
           ROOT/'configs/stage2-visual-plan.json',ROOT/'configs/evaluation-protocol-v1.json']
    files += [ROOT/'scripts'/name for name in ['train_stage2_grpo.py','stage2_runtime.py','stage2_reward.py',
               'evaluation_core.py','evaluate_stage2.py','train_stage2_sft_control.py',
               'freeze_stage2_gate.py','run_stage2_experiment.py','stage2_budget.py',
               'audit_stage2_training.py','verify_stage2_checkpoint.py','verify_stage2_reload.py',
               'freeze_stage2_models.py','summarize_stage2.py','export_stage2_visuals.py']]
    result=dict(passed=True,preflight=args.preflight,
                evidence={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                projected_optimization_gpu_hours=projected_optimization,
                projected_formal_reserved_gpu_hours=projected_formal_reserved,
                measured_seconds_per_candidate_including_updates=candidate_seconds,
                estimated_serial_seconds_per_example=serial_seconds,
                serial_evaluation_estimate=dict(preflight_mean_output_tokens=mean_tokens,
                    measured_serial_tokens_per_second=serial['tokens_per_second'],margin=1.25,
                    scope='Single training-prompt speed extrapolated to full preflight mean length; runtime budgets still enforced'),
                estimated_validation_wall_hours=validation_hours,
                generation_resource_decision=dict(concurrency=selected['concurrency'],
                    measured_token_throughput_speedup=measured_speedup,credited_speedup=conservative_speedup,
                    unchanged_runtime_fraction=.25,scope='Resource-only change; one-prompt estimate,verify actual formal throughput'),
                reserve_other_gpu_hours=remaining_for_other,resources_at_freeze=resources,
                formal_settings=dict(train_prompts=512,candidates_per_prompt=4,updates=128,
                    base_model=str(ROOT/'models/Qwen2.5-VL-7B-Instruct'),
                    adapter_start=str(ROOT/'outputs/stage1-A-image-edit-3k-v1'),
                    quantization_bits=4,quantization_type='nf4',double_quantization=True,compute_dtype='bfloat16',
                    lora_rank=8,lora_alpha=16,lora_dropout=0.,target_modules=['q_proj','v_proj'],
                    freeze_vision=True,image_max_pixels=262144,
                    candidate_microbatch=1,gradient_accumulation=16,learning_rate=1e-6,warmup_ratio=.05,
                    generation_concurrency=selected['concurrency'],
                    warmup_steps=7,scheduler='cosine',optimizer='adamw_torch',weight_decay=0.,
                    temperature=.8,top_p=.9,top_k=0,beta=.01,loss_type='grpo',scale_rewards='group',
                    epsilon=.2,num_iterations=1,max_grad_norm=1.,max_completion_length=512,
                    seed=20261101,eval_save_steps=[64,128],selection='full validation mean edited image L1',
                    reference='frozen original A adapter',generation_backend='transformers',
                    max_formal_gpu_hours=12,total_experiment_gpu_hours=16),
                sft_control=dict(start='A',rows=512,microbatch=1,gradient_accumulation=4,updates=128,
                                 repeats_per_row=1,learning_rate=1e-6,selection='final step128; no test selection'))
    out=ROOT/'configs/stage2-formal-gate.json';assert not out.exists()
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

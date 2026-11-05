"""Fresh-process adapter reload and preflight gate; no optimizer steps."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone
import traceback
import torch
from transformers.feature_extraction_utils import BatchFeature
from trl.models.utils import peft_module_casting_to_bf16
from stage2_runtime import ROOT, load_model, adapter_state, tensor_digest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--formal',action='store_true')
    parser.add_argument('--sft',action='store_true')
    parser.add_argument('--attempt',type=int,default=1)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2'
    assert int(subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.used',
                                       '--format=csv,noheader,nounits'],text=True))<512
    out=ROOT/'outputs/stage2-A-edit-grpo-v1'/args.run_name
    training=json.loads((out/'training-verification.json').read_text())
    assert not (args.formal and args.sft)
    assert training['step']==(128 if args.formal or args.sft else 8)
    if args.sft:
        assert training['seen_samples']==512 and training['supervised_tokens']==training['expected_supervised_tokens']
    else:
        assert training['candidates']==(2048 if args.formal else 128)
    assert not (out/'reload-verification.json').exists()
    started=time.monotonic()
    record_dir=out/f'reload-attempt-{args.attempt}'
    record_dir.mkdir(exist_ok=False)
    status=dict(status='running',pid=os.getpid(),gpu=2,started_at=datetime.now(timezone.utc).isoformat())
    def save_status():
        tmp=record_dir/'status.tmp'
        tmp.write_text(json.dumps(status,indent=2)+'\n')
        tmp.replace(record_dir/'status.json')
    save_status()
    try:
        torch.set_num_threads(4)
        model=load_model(out/'final',trainable=True)
        peft_module_casting_to_bf16(model)
        assert tensor_digest(adapter_state(model))==training['final_adapter_digest']
        # Older own-run probes retained the processor's BatchFeature wrapper.
        # Allow only this known container while keeping weights-only loading.
        with torch.serialization.safe_globals([BatchFeature]):
            probe=torch.load(out/'reload-probe.pt',weights_only=True)
        inputs={k:v.to('cuda') for k,v in probe['input'].items()}
        expected=torch.load(out/'reload-expected-logits.pt',weights_only=True)
        model.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(**inputs,use_cache=False).logits[:,-1].float().cpu()
        assert torch.isfinite(logits).all()
        difference=float((logits-expected).abs().max())
        assert torch.equal(logits,expected), difference
        peak=max(int(line.split(',')[1]) for line in (out/'gpu.csv').read_text().splitlines())/1024
        groups=training.get('groups',[])
        report=dict(status='passed',adapter_tensors_exact=True,forward_exact=True,
                    max_logit_difference=difference,training_updates=0,sample_id=probe['sample_id'],
                    elapsed_seconds=time.monotonic()-started,gpu_hours=(time.monotonic()-started)/3600,
                    training_device_peak_gib=peak,
                    diverse_output_groups=sum(g['distinct_outputs']>1 for g in groups),
                    informative_reward_groups=sum(g['distinct_rewards']>1 for g in groups),
                    formal_gate_passed=not args.sft and peak<=21 and sum(g['distinct_rewards']>1 for g in groups)>=len(groups)/2,
                    gate_note='At least half the groups must have distinct rewards; otherwise diagnose signal before formal run.')
        (out/'reload-verification.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        status['status']='completed'
    except BaseException as exc:
        status.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        status.update(updated_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
                      gpu_hours=(time.monotonic()-started)/3600)
        save_status()



if __name__=='__main__':
    main()

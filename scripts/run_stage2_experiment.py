"""Run the accepted formal experiment once, retaining every failure for inspection.

Requires a separately frozen formal gate. Ends before qualitative review/reporting.
No test model selection, automatic restart, or overwrite of existing runs.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from stage2_budget import ROOT, ledger

OUT=ROOT/'outputs/stage2-A-edit-grpo-v1'
LOG=ROOT/'logs/stage2-A-edit-grpo-v1'
PYTHON=ROOT/'envs/iea-rl/bin/python'


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    gate=read(ROOT/'configs/stage2-formal-gate.json')
    assert gate['passed']
    for path,digest in gate['evidence'].items():
        assert sha(path)==digest,path
    assert not (OUT/'formal-v1').exists() and not (OUT/'sft-control').exists()
    assert not (ROOT/'configs/stage2-selected-models.json').exists()
    statepath=LOG/'experiment-pipeline.json'
    assert not statepath.exists(), 'Inspect existing pipeline; never launch twice'
    state=dict(status='starting',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),completed=[])
    def save(**kwargs):
        state.update(updated_at=datetime.now(timezone.utc).isoformat(),**kwargs)
        temp=statepath.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(statepath)
    def require_budget():
        resources=ledger()
        assert resources['budget_safe_to_launch'],resources
        return resources['remaining_gpu_hours']*3600
    def wait_gpu(gpu):
        deadline=time.monotonic()+1800
        while int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used',
                                          '--format=csv,noheader,nounits'],text=True))>=512:
            assert time.monotonic()<deadline,f'GPU{gpu} unavailable; jobs left untouched'
            save(status='waiting_for_gpu',gpu=gpu);time.sleep(10)
    def run(label,script,arguments=(),gpu=None):
        if gpu is not None:
            require_budget();wait_gpu(gpu)
        env=dict(os.environ,TOKENIZERS_PARALLELISM='false')
        if gpu is not None: env['CUDA_VISIBLE_DEVICES']=str(gpu)
        command=[str(PYTHON),'-u',str(ROOT/'scripts'/script),*map(str,arguments)]
        logfile=LOG/f'pipeline-{label}.log'
        with logfile.open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            save(status='running',phase=label,child_pid=child.pid,command=command,log=str(logfile))
            while child.poll() is None:
                time.sleep(10)
                save(child_pid=child.pid)
            assert child.returncode==0,f'{label} exited {child.returncode}; inspect {logfile}; no automatic restart'
        state['completed'].append(label)
        save(child_pid=None)
    def checkpoint(run_name,step):
        folder=OUT/run_name
        output=subprocess.check_output([str(PYTHON),str(ROOT/'scripts/verify_stage2_checkpoint.py'),
                                         str(folder/f'checkpoint-{step}'),'--step',str(step)],cwd=ROOT,text=True)
        report=json.loads(output);assert report['passed']
        target=folder/f'checkpoint-{step}-audit.json';assert not target.exists()
        target.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        run('formal','train_stage2_grpo.py',['--formal','--run-name','formal-v1'],gpu=2)
        formal=OUT/'formal-v1'
        assert read(formal/'training-verification.json')['step']==128
        run('formal-candidates','audit_stage2_training.py',['--run-name','formal-v1','--formal'])
        for step in [64,128]: checkpoint('formal-v1',step)
        run('formal-reload','verify_stage2_reload.py',['--run-name','formal-v1','--formal'],gpu=2)
        # Protect the measured final-test reserve before spending the SFT allowance.
        test_reserve=900*gate['estimated_serial_seconds_per_example']
        allowance=min(1800,require_budget()-test_reserve-360)
        assert allowance>600,'Insufficient remaining reserve for SFT and final tests'
        run('sft','train_stage2_sft_control.py',['--max-gpu-seconds',allowance],gpu=0)
        sft=OUT/'sft-control'
        assert read(sft/'training-verification.json')['step']==128
        for step in [64,128]: checkpoint('sft-control',step)
        run('sft-reload','verify_stage2_reload.py',['--run-name','sft-control','--sft'],gpu=2)
        run('model-freeze','freeze_stage2_models.py')
        models=read(ROOT/'configs/stage2-selected-models.json')['models']
        # Deterministic evaluations are intentionally serial: resource accounting is
        # straightforward and both GPUs remain available between stages.
        for model in ['A','SFT','GRPO']:
            for split,count in [('test_new',100),('test_regression',200)]:
                estimate=count*gate['estimated_serial_seconds_per_example']+120
                allowance=min(estimate,require_budget()-120)
                assert allowance>300,'Insufficient remaining evaluation budget'
                run(f'{split}-{model}','evaluate_stage2.py',
                    ['--adapter',models[model]['adapter'],'--name',model,'--split',split,
                     '--gpu','0','--max-gpu-seconds',allowance],gpu=0)
                folder=OUT/'evaluation'/f'{split}-{model}'
                assert read(folder/'status.json')['status']=='completed'
                assert read(folder/'summary.json')['rows']==count
        run('statistics','summarize_stage2.py')
        run('anonymous-bundle','export_stage2_visuals.py')
        save(status='numerical_experiment_complete_pending_visual_review_and_report',
             resources=ledger(),remaining=['Inspect30anonymouscases and save90observations beforedecoding',
                                         'Create curves/report,verify all artifacts,download archive,final requirement audit'])
    except BaseException as exc:
        save(status='failed',error=str(exc),traceback=traceback.format_exc(),resources=ledger())
        raise


if __name__=='__main__':
    main()

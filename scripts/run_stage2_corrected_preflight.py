"""Queue one corrected preflight after the existing run exits; never resume it."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

ROOT = Path('/mnt/sdc/zhc/IEA')
LOG = ROOT/'logs/stage2-A-edit-grpo-v1'
OUT = ROOT/'outputs/stage2-A-edit-grpo-v1'
PYTHON = ROOT/'envs/iea-rl/bin/python'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    statefile=LOG/'corrected-preflight-queue.json'
    assert not statefile.exists(), 'Existing queue must be inspected, never duplicated'
    state=dict(status='waiting',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat())
    def save(**kwargs):
        state.update(updated_at=datetime.now(timezone.utc).isoformat(),**kwargs)
        tmp=statefile.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(statefile)
    save()
    try:
        staging=ROOT/'scripts/stage2-corrected-preflight'
        manifest=read(staging/'manifest.json')
        assert not (OUT/'preflight-v8').exists()
        for name,digest in manifest.items():
            assert sha(staging/name)==digest,name
        deadline=time.monotonic()+1800
        previous=OUT/'preflight-v7'
        diagnostic=OUT/'padding-diagnostic-fixed-all'
        while True:
            old=read(previous/'status.json')
            proc=Path('/proc')/str(old['pid'])/'cmdline'
            cmd=proc.read_bytes() if proc.exists() else b''
            alive=b'train_stage2_grpo.py' in cmd
            ds=read(diagnostic/'status.json')
            if ds['status']=='failed':
                raise RuntimeError('Full padding diagnostic failed; no corrected training launched')
            if not alive and ds['status']=='completed':
                break
            assert time.monotonic()<deadline, 'Queue timeout; existing jobs left untouched'
            save(previous_step=old.get('step'),previous_process_alive=alive,diagnostic_status=ds['status'])
            time.sleep(10)
        assert old['status']=='awaiting_independent_reload' and old['step']==8,old
        check=read(diagnostic/'result.json')
        assert check['passed'] and check['prompts']==32 and check['concurrency']==4
        # Retain all v7 artifacts and explicitly disqualify it from formal acceptance.
        rejected=previous/'acceptance-rejected.json'
        assert not rejected.exists()
        rejected.write_text(json.dumps(dict(accepted=False,reason='Left-padded cached generation changes output',
            diagnostic=str(OUT/'padding-diagnostic/result.json'),training_start_for_successor='Original A, never v7'),indent=2)+'\n')
        while int(subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.used',
                                          '--format=csv,noheader,nounits'],text=True))>=512:
            assert time.monotonic()<deadline, 'GPU2 unavailable; no job touched'
            save(status='waiting_for_gpu2');time.sleep(10)
        # Copy only reviewed, hash-locked implementation files after v7 has exited.
        backup=LOG/'source-before-preflight-v8';backup.mkdir(exist_ok=False)
        for name,digest in manifest.items():
            assert sha(staging/name)==digest
            target=ROOT/'scripts'/name
            if target.exists():
                shutil.copy2(target,backup/name)
            shutil.copy2(staging/name,target)
            assert sha(target)==digest
        resource=json.loads(subprocess.check_output([str(PYTHON),str(ROOT/'scripts/stage2_budget.py')],text=True))
        assert resource['budget_safe_to_launch'] and resource['remaining_gpu_hours']>12
        command=[str(PYTHON),'-u',str(ROOT/'scripts/train_stage2_grpo.py'),
                 '--preflight','--run-name','preflight-v8','--generation-concurrency','4']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',TOKENIZERS_PARALLELISM='false')
        with (LOG/'preflight-v8.log').open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            launch=dict(pid=child.pid,command=command,source_hashes=manifest,
                        started_at=datetime.now(timezone.utc).isoformat(),origin='Original A')
            (LOG/'preflight-v8-launch.json').write_text(json.dumps(launch,indent=2)+'\n')
            save(status='corrected_preflight_running',trainer_pid=child.pid)
            code=child.wait()
        assert code==0,f'Corrected preflight exit {code}; evidence retained, no restart'
        run=OUT/'preflight-v8'
        save(status='auditing')
        subprocess.run([str(PYTHON),str(ROOT/'scripts/audit_stage2_training.py'),
                        '--run-name','preflight-v8'],cwd=ROOT,check=True)
        result=subprocess.check_output([str(PYTHON),str(ROOT/'scripts/verify_stage2_checkpoint.py'),
                                         str(run/'checkpoint-8'),'--step','8'],cwd=ROOT,text=True)
        report=json.loads(result);assert report['passed']
        (run/'checkpoint-audit.json').write_text(json.dumps(report,indent=2)+'\n')
        with (LOG/'preflight-v8-reload.log').open('x') as log:
            subprocess.run([str(PYTHON),'-u',str(ROOT/'scripts/verify_stage2_reload.py'),
                            '--run-name','preflight-v8'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        reload=read(run/'reload-verification.json')
        assert reload['formal_gate_passed'], 'Reload succeeded but diversity/memory gate failed'
        save(status='preflight_accepted_pending_formal_freeze')
    except BaseException as exc:
        save(status='failed',error=str(exc),traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    main()

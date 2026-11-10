"""Wait on the actual queue process and prepare a completed epoch, without grading."""
import argparse,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];EXP=ROOT/'experiments/functional-v19-20261107'

def read(path):return json.loads(path.read_text())
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',type=int,choices=(1,2),required=True);args=parser.parse_args()
    mid=f'photometric-bridge-e{args.epoch}';deadline=time.monotonic()+3600
    while True:
        assert time.monotonic()<deadline,'Review preparation deadline; preserve queue/training and inspect'
        queue=read(EXP/'evaluation-queue.json');job=next((j for j in queue['jobs'] if j['model']==mid),None)
        if job and 'returncode' in job:
            assert job['returncode']==0,job
            break
        proc=Path('/proc')/str(queue['pid'])
        assert proc.exists() and b'functional_v19.evaluation_queue' in (proc/'cmdline').read_bytes(),'Queue process is no longer live before terminal result; inspect without restarting training'
        time.sleep(5)
    binding=EXP/f'epoch{args.epoch}-review-binding.json'
    assert not binding.exists(),'Existing packet preserved; do not prepare again'
    command=[str(ROOT/'envs/iea-rl/bin/python'),'-m','functional_v19.prepare_epoch_review','--epoch',str(args.epoch)]
    subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT/'scripts'),CUDA_VISIBLE_DEVICES=''),check=True)
    print('Prepared actual terminal-success epoch packet',args.epoch,'; no scores, judgments or training changes.',flush=True)

if __name__=='__main__':main()

"""Uniform local generation + fixed editor; JSONL output is resumable and auditable."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import time
import traceback

ROOT = Path('/mnt/sdc/zhc/IEA')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['Base', 'A', 'B', 'Official', 'BaseSchema'], required=True)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], required=True)
    parser.add_argument('--gpu', choices=['0', '2', '3'], required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--run-name', default='full')
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == args.gpu, 'Activate the requested IEA_GPU before running.'
    assert args.limit >= 0 and args.run_name.replace('-', '').replace('_', '').isalnum()
    occupants = subprocess.check_output(['nvidia-smi', '-i', args.gpu, '--query-compute-apps=pid',
                                         '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    assert not [pid for pid in occupants if pid.strip() != str(os.getpid())], 'Requested GPU is occupied.'
    protocol_path = ROOT / 'configs/evaluation-protocol-v1.json'
    protocol = json.loads(protocol_path.read_text())
    if args.split != 'train':
        assert protocol['status'] == 'frozen', 'Only training probes are permitted before protocol freezing.'
    if args.split == 'test':
        assert args.limit == 0, 'The final test must include every frozen test row.'
        model_lock = json.loads((ROOT / 'configs/evaluation-models-v1.json').read_text())
        assert model_lock['all_selected_models_verified']
        for item in model_lock['models'][args.model]['identity_files']:
            assert sha(Path(item['path'])) == item['sha256'], item['path']
        for item in model_lock['models'][args.model]['weight_shards']:
            stat = Path(item['path']).stat()
            assert stat.st_size == item['bytes'] and stat.st_mtime_ns == item['mtime_ns'], item['path']
    for path, digest in protocol.get('code_hashes', {}).items():
        assert sha(Path(path)) == digest, path
    from evaluation_core import Scorer, parse_parameters, sample_seed
    import torch
    from transformers import set_seed
    from llamafactory.chat import ChatModel

    source = Path(protocol['split_directory']) / f'{args.split}.json'
    manifest = json.loads((source.parent / 'manifest.json').read_text())
    assert sha(source) == manifest['files'][args.split]['sha256']
    rows = json.loads(source.read_text())
    if args.limit:
        rows = rows[:args.limit]
    out = ROOT / f'outputs/evaluation-v1/{args.split}-{args.model}-{args.run_name}'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = dict(model=args.model, split=args.split, selected_ids=[r['id'] for r in rows],
                    source_sha256=sha(source), protocol_sha256=sha(protocol_path),
                    worker_sha256=sha(Path(__file__)), core_sha256=sha(Path(__file__).with_name('evaluation_core.py')))
    meta_path = out / 'run.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        assert meta['identity'] == identity, 'Cannot resume with altered inputs, protocol or code.'
    else:
        meta = dict(identity=identity, started_at=stamp(), pid=os.getpid(), gpu=args.gpu)
        write(meta_path, meta)
        for path in [Path(__file__), Path(__file__).with_name('evaluation_core.py'), protocol_path]:
            shutil.copy2(path, out / path.name)
    predictions = out / 'predictions.jsonl'
    done = []
    if predictions.exists():
        lines = predictions.read_text().splitlines()
        for line in lines:
            done.append(json.loads(line))  # A partial record requires explicit repair, never silent deletion.
    ids = [r['id'] for r in done]
    assert ids == [r['id'] for r in rows[:len(done)]], 'Unexpected or duplicate prediction IDs.'
    if len(done) == len(rows):
        print('ALREADY_COMPLETE', out, flush=True)
        return
    free = int(subprocess.check_output(['nvidia-smi', '-i', args.gpu, '--query-gpu=memory.free',
                                        '--format=csv,noheader,nounits'], text=True).strip())
    assert free >= 14000, f'GPU {args.gpu} free {free} MiB is below the preflight threshold.'
    started = time.monotonic()
    status = dict(status='loading', pid=os.getpid(), gpu=args.gpu, model=args.model, split=args.split,
                  completed=len(done), total=len(rows), updated_at=stamp(), output_dir=str(out))
    write(out / 'status.json', status)
    model_args = {**protocol['model_common'], **protocol['models'][args.model], **protocol['generation']}
    # These generating arguments are call-time controls, not constructor fields.
    model_args.pop('num_return_sequences', None)
    set_seed(protocol['seed'])
    try:
        chat = ChatModel(model_args)
        scorer = Scorer(ROOT, protocol)
    except BaseException as exc:
        status.update(status='failed', phase='model_loading', updated_at=stamp(), error=f'{type(exc).__name__}:{exc}')
        write(out / 'status.json', status)
        raise
    status.update(status='running', model_loaded_at=stamp())
    write(out / 'status.json', status)
    try:
        with predictions.open('a', buffering=1) as stream:
            for row in rows[len(done):]:
                begin = time.monotonic()
                seed = sample_seed(row['id'], protocol['seed'])
                set_seed(seed)
                system = [x['content'] for x in row['conversations'] if x['role'] == 'system']
                messages = [dict(role=x['role'], content=x['content']) for x in row['conversations'] if x['role'] == 'user']
                assert len(system) == 1 and len(messages) == 1 and len(row['images']) == 1
                system_prompt = system[0] + protocol.get('system_suffix_by_model', {}).get(args.model, '')
                assert row['images'][0] == row['extra_info']['origin_image_path']
                record = dict(id=row['id'], group_id=row['group_id'], model=args.model,
                              instruction=row['extra_info']['instruction'],
                              instruction_type=row['extra_info']['inst_type'],
                              input_image=row['images'][0], reference_image=row['extra_info']['reference_image_path'],
                              target_parameters=row['extra_info']['label'],
                              rare_tool_subgroup=any(row['extra_info']['label'].get(k, 0) != 0 for k in ('exposure', 'sharpness')),
                              seed=seed, inference_error=None)
                oom = False
                try:
                    answer = chat.chat(messages, system=system_prompt, images=row['images'], **protocol['generation'])[0]
                    record.update(asdict(answer))
                except Exception as exc:
                    record.update(response_text='', response_length=0, prompt_length=None, finish_reason='error',
                                  inference_error=f'{type(exc).__name__}:{exc}', traceback=traceback.format_exc())
                    oom = isinstance(exc, torch.cuda.OutOfMemoryError)
                record['generation_seconds'] = time.monotonic() - begin
                parsed = parse_parameters(record['response_text'], protocol)
                record['parsing'] = parsed
                file_id = hashlib.sha256(row['id'].encode()).hexdigest()[:20]
                record['scoring'] = scorer.score(row, parsed, out / 'edited' / f'{file_id}.png')
                record['total_seconds'] = time.monotonic() - begin
                record['completed_at'] = stamp()
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                done.append(record)
                status.update(completed=len(done), updated_at=stamp(), last_id=row['id'],
                              valid_outputs=sum(x['parsing']['valid'] for x in done),
                              editor_successes=sum(x['scoring']['editor_success'] for x in done),
                              inference_errors=sum(x['inference_error'] is not None for x in done),
                              elapsed_seconds=round(time.monotonic() - started, 3),
                              cuda_peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                              cuda_peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
                write(out / 'status.json', status)
                print(json.dumps(status), flush=True)
                if oom:
                    raise RuntimeError('CUDA OOM recorded; stop to inspect instead of changing evaluation settings.')
        status.update(status='completed', updated_at=stamp(), completed_at=stamp(),
                      parse_error_counts=dict(Counter(x['parsing']['error'] for x in done if x['parsing']['error'])))
        write(out / 'status.json', status)
    except BaseException as exc:
        status.update(status='failed', updated_at=stamp(), error=f'{type(exc).__name__}:{exc}')
        write(out / 'status.json', status)
        raise
    print('EVALUATION_COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()

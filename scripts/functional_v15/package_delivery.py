"""Self-auditing snapshot; deduplicate identical bytes without dropping path references."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
from functional_v1.common import ROOT, sha, read, write, stamp


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk:=handle.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def main():
    out=ROOT/'archives/functional-v15-20261106'
    assert (ROOT/'experiments/functional-v15-20261106/development-selection.json').exists()
    assert all(read(p)['status'] == 'completed' for p in (ROOT/'experiments/functional-v15-20261106/runs').glob('*/status.json'))
    out.mkdir(parents=True,exist_ok=True)
    stage=out/'complete-snapshot'
    assert not stage.exists(), 'Keep previous delivered snapshots immutable'
    files=set()
    for name in ('README.md','实验记录与结论.md','原理与复用指南.md','models/README.md','models/catalog.json','GITHUB_INDEX.md','docs/PROJECT_STATUS.md','docs/REPOSITORY_MAP.md','docs/ARTIFACTS.json'):
        files.add(ROOT/name)
    # Restore verified prior archives for old source images and parent adapters.
    for directory in (ROOT/'experiments/functional-v15-20261106',ROOT/'data/training/functional-v15-20261106',
                      ROOT/'experiments/functional-v16-20261106',ROOT/'experiments/functional-v17-20261107',
                      *(ROOT/'scripts'/f'functional_v{i}' for i in range(1,18)),
                      ROOT/'models/from-functional-v11/sft-shadow-correction'):
        for p in directory.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts and p.suffix!='.lock':files.add(p)
    for relative in ('experiments/functional-v1-20261104/resources.json',
                     'experiments/functional-v1-20261104/development-selection.json',
                     'experiments/functional-v2-20261104/development-selection.json',
                     'data/stage1/ready/GIER-Edit/sft_train.json','data/stage1/ready/GIER-Edit/sft_test.json',
                     'data/stage1/splits/image-edit-3k-v1/group-assignments.json','data/sources/gier/GIER.json',
                     'archives/functional-v1-20261104/receipt.json','archives/functional-v2-20261104/receipt.json','archives/functional-v3-v4-20261105/receipt.json','archives/functional-v5-v6-20261105/receipt.json','archives/functional-v7-20261105/receipt.json','archives/functional-v8-v9-20261106/receipt.json','archives/functional-v10-20261106/receipt.json','archives/functional-v11-20261106/complete-receipt.json',
                     'archives/functional-v12-20261106/complete-receipt.json','archives/functional-v13-20261106/complete-receipt.json','archives/functional-v14-20261106/complete-receipt.json',
                     'configs/evaluation-weight-integrity-v1.json'):
        files.add(ROOT/relative)
    protocol=read(ROOT/'experiments/functional-v15-20261106/training-protocol.json')
    assert all(sha(ROOT/p)==h for p,h in protocol['code_hashes'].items())
    cache=read(ROOT/'experiments/functional-v15-20261106/batch-cache.json')
    assert cache['cache_path']=='experiments/functional-v15-20261106/batches.pt'
    assert cache['cache_sha256']==sha(ROOT/cache['cache_path'])
    # Collect exact verified request IDs, including the actually switched R14 model.
    # A historical model-name prefix is not a safe scope boundary for a new round.
    expected_requests={r['request_id'] for r in read(ROOT/'experiments/functional-v15-20261106/demo/http-integration.json')['checks']}
    expected_requests.update(r['request_id'] for r in read(ROOT/'experiments/functional-v15-20261106/demo/browser-integration.json')['requests'])
    assert len(expected_requests)==16
    assert len(read(ROOT/'experiments/functional-v15-20261106/demo/agent-feedback-records.json')['records'])==2
    for rid in expected_requests:
        folder=ROOT/'experiments/functional-v1-20261104/demo/requests'/rid
        record=read(folder/'request.json')
        assert record.get('source') in ('agent_round15_HTTP_test_not_user_feedback','agent_round15_browser_test_not_user_feedback')
        assert record['training_use'] is False
        assert record['model_id'] in ('functional-v15-shadow-correction-20261127','functional-v14-visual-late4-20261127')
        files.update(q for q in folder.iterdir() if q.is_file())
        assert folder/'request.json' in files and folder/'input.png' in files
        if record['status']=='executed':assert folder/'output.png' in files
        else:assert not (folder/'output.png').exists()
    stage.mkdir()
    unique={};manifest=[]
    for source in sorted(files):
        relative=source.relative_to(ROOT);target=stage/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        # Copy the changing logs/ledger as a point-in-time snapshot before hashing.
        shutil.copy2(source,target)
        digest=sha(target)
        if digest in unique:
            target.unlink();os.link(unique[digest],target)
        else:unique[digest]=target
        manifest.append({'path':str(relative),'bytes':target.stat().st_size,'sha256':digest})
    write(stage/'MANIFEST.json',{'created_at':stamp(),'files':manifest,
        'not_included':['Full original/author weights, Python environments, whole GIER pool, prior-round source images and parent adapters (restore eleven verified dependencies), non-agent trial uploads and future work. Includes actual R15 data/cache, all label proposals and failures, image/editor control, training checkpoints/optimizer, AI review, trial, and completed R16 precision/R17 TRAIN-resolution diagnostics. R16/R17 have no policy training or models.'],
        'dependencies': [dict(archive='archives/functional-v1-20261104/functional-v1-20261104.tar.gz',sha256='5605fa8532feb37458ffa9f6624a7ef36521e6d34011fa2b8ea51de7165c2ca5'),dict(archive='archives/functional-v2-20261104/functional-v2-20261104.tar.gz',sha256='00fd797acb73b59b8f42c1aa49f1f24aa4edb26e9a76c080755795f0f1506b11'),dict(archive='archives/functional-v3-v4-20261105/functional-v3-v4-20261105.tar.gz',sha256='8ca75376b83ab70321ad41def1b97bd2c46d883ab080cabf1b0753ca91ed8780'),dict(archive='archives/functional-v5-v6-20261105/functional-v5-v6-20261105.tar.gz',sha256='2a7e513a3262bddc5df26d3ab1c42081f68519f905ccbebc0fe551801ffe921b'),dict(archive='archives/functional-v7-20261105/functional-v7-20261105.tar.gz',sha256='fcbee5bfe3a10b7568d73f34dcf634fd9c7001fd5a978b8ba41523f2bef97fa7'),dict(archive='archives/functional-v8-v9-20261106/functional-v8-v9-20261106.tar.gz',sha256='ef5192be6ec597dd1ee21bc9d47290f5f47c9af5ac5c568971402a5cfc83114d'),dict(archive='archives/functional-v10-20261106/functional-v10-20261106.tar.gz',sha256='2de9ad0e438997f206acb8270ddd94ed41d8233c05566b61f5366b5f9a996f27'),dict(archive='archives/functional-v11-20261106/functional-v11-20261106-complete.tar.gz',sha256='8a9f63f1e49ee8b7d12aba5d0485649d1353165ef675322cb5217072643ffee3'),dict(archive='archives/functional-v12-20261106/functional-v12-20261106-complete.tar.gz',sha256='4c8f3cf87e13f2f4653f5661aa78037220e6d9bc49ec6543c8e21211c302a495'),dict(archive='archives/functional-v13-20261106/functional-v13-20261106-complete.tar.gz',sha256='0e9dee0b406181fcd5b3acaad8532ff22a025f7f9b8515a2fac603a1dd52927c'),dict(archive='archives/functional-v14-20261106/functional-v14-20261106-complete.tar.gz',sha256='cdd0f54a096fa2052a3bbe80d63c29fcb81d64009268e95fc6c83b6d9fca12d3')],
        'actual_cache':cache,
        'deduplication':'Identical file bytes retain every original path through tar hard links.'})
    archive=out/'functional-v15-20261106-complete.tar.gz'
    with archive.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',compresslevel=1,fileobj=raw,mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w') as tf:
        tf.add(stage,arcname='functional-v15-20261106')
    # Verify every regular file and hard link by streaming from the actual archive.
    expected={r['path']:r for r in manifest};verified=[];stored={}
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf:
            relative=str(Path(member.name).relative_to('functional-v15-20261106'))
            if relative not in expected:continue
            if member.islnk():
                digest,size=stored[member.linkname]
            else:
                f=tf.extractfile(member);h=__import__('hashlib').sha256();size=0
                while chunk:=f.read(1024*1024):h.update(chunk);size+=len(chunk)
                digest=h.hexdigest()
            assert digest==expected[relative]['sha256'] and size==expected[relative]['bytes']
            stored[member.name]=(digest,size)
            verified.append(relative)
    assert len(verified)==len(manifest)
    write(out/'complete-receipt.json',{'status':'archive_verified','scope':'Complete round15 shadow-label/editor correction and separate precision/resolution diagnostics at the recorded result state; training and AI review are not independent user usability certification; later work excluded','created_at':stamp(),'archive':str(archive.relative_to(ROOT)),
        'sha256':sha(archive),'bytes':archive.stat().st_size,'verified_files':len(verified),
        'unique_file_contents':len(unique),'source_logical_bytes':sum(r['bytes'] for r in manifest),
        'manifest_sha256':sha(stage/'MANIFEST.json')})
    print(json.dumps(read(out/'complete-receipt.json')))


if __name__=='__main__':main()

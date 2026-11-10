"""Self-auditing snapshot; deduplicate identical bytes without dropping path references."""
import gzip
import json
import os
from pathlib import Path
import shutil
import tarfile
from functional_v1.common import ROOT, sha, read, write, stamp


def main():
    out=ROOT/'archives/functional-v8-v9-20261106'
    assert (ROOT/'experiments/functional-v9-20261106/development-selection.json').exists()
    assert all(read(p)['status'] == 'completed' for p in (ROOT/'experiments/functional-v9-20261106/runs').glob('*/status.json'))
    out.mkdir(parents=True,exist_ok=True)
    stage=out/'snapshot'
    assert not stage.exists(), 'Keep previous delivered snapshots immutable'
    stage.mkdir()
    files=set()
    for name in ('README.md','实验记录与结论.md','原理与复用指南.md','models/README.md','models/catalog.json'):
        files.add(ROOT/name)
    # Restore verified prior archives for old source images and parent adapters.
    for directory in (ROOT/'experiments/functional-v8-20261106', ROOT/'experiments/functional-v9-20261106',
                      ROOT/'data/training/functional-v8-20261106', ROOT/'data/training/functional-v9-20261106',
                      *(ROOT/'scripts'/f'functional_v{i}' for i in range(1,10)),
                      ROOT/'models/from-base/sft-fresh-outcome', ROOT/'models/from-author-iea/sft-fresh-outcome',
                      ROOT/'models/external/EditScore-7B', ROOT/'third_party/EditScore'):
        for p in directory.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts and p.suffix!='.lock':files.add(p)
    for relative in ('experiments/functional-v1-20261104/resources.json',
                     'experiments/functional-v1-20261104/development-selection.json',
                     'experiments/functional-v2-20261104/development-selection.json',
                     'data/stage1/ready/GIER-Edit/sft_train.json','data/stage1/ready/GIER-Edit/sft_test.json',
                     'data/stage1/splits/image-edit-3k-v1/group-assignments.json','data/sources/gier/GIER.json',
                     'archives/functional-v1-20261104/receipt.json','archives/functional-v2-20261104/receipt.json','archives/functional-v3-v4-20261105/receipt.json','archives/functional-v5-v6-20261105/receipt.json','archives/functional-v7-20261105/receipt.json',
                     'configs/evaluation-weight-integrity-v1.json'):
        files.add(ROOT/relative)
    # Preserve only explicitly labelled agent browser/API uploads, never user trial data.
    for p in (ROOT/'experiments/functional-v1-20261104/demo/requests').glob('*/request.json'):
        if read(p).get('source') not in ('agent_round9_HTTP_test_not_user_feedback','agent_round9_browser_test_not_user_feedback'):continue
        if read(p).get('model_id','').startswith('functional-v9-'):
            files.update(q for q in p.parent.iterdir() if q.is_file())
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
        'not_included':['Original Qwen and author IEA full weights, Python environments, prior-round source images and prior adapters (restore all five verified prior archives in order), non-agent trial uploads and later experiments. Includes complete round8 failed critic calibration, its pinned external adapter/source, and round9 training/evaluation/trial artifacts.'],
        'dependencies': [dict(archive='archives/functional-v1-20261104/functional-v1-20261104.tar.gz',sha256='5605fa8532feb37458ffa9f6624a7ef36521e6d34011fa2b8ea51de7165c2ca5'),dict(archive='archives/functional-v2-20261104/functional-v2-20261104.tar.gz',sha256='00fd797acb73b59b8f42c1aa49f1f24aa4edb26e9a76c080755795f0f1506b11'),dict(archive='archives/functional-v3-v4-20261105/functional-v3-v4-20261105.tar.gz',sha256='8ca75376b83ab70321ad41def1b97bd2c46d883ab080cabf1b0753ca91ed8780'),dict(archive='archives/functional-v5-v6-20261105/functional-v5-v6-20261105.tar.gz',sha256='2a7e513a3262bddc5df26d3ab1c42081f68519f905ccbebc0fe551801ffe921b'),dict(archive='archives/functional-v7-20261105/functional-v7-20261105.tar.gz',sha256='fcbee5bfe3a10b7568d73f34dcf634fd9c7001fd5a978b8ba41523f2bef97fa7')],
        'deduplication':'Identical file bytes retain every original path through tar hard links.'})
    archive=out/'functional-v8-v9-20261106.tar.gz'
    with archive.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',compresslevel=1,fileobj=raw,mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w') as tf:
        tf.add(stage,arcname='functional-v8-v9-20261106')
    # Verify every regular file and hard link by streaming from the actual archive.
    expected={r['path']:r for r in manifest};verified=[];stored={}
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf:
            relative=str(Path(member.name).relative_to('functional-v8-v9-20261106'))
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
    write(out/'receipt.json',{'status':'archive_verified','scope':'Complete round8 critic calibration and round9 origin comparison at the recorded result state; training and AI review are not independent user usability certification; later work excluded','created_at':stamp(),'archive':str(archive.relative_to(ROOT)),
        'sha256':sha(archive),'bytes':archive.stat().st_size,'verified_files':len(verified),
        'unique_file_contents':len(unique),'source_logical_bytes':sum(r['bytes'] for r in manifest),
        'manifest_sha256':sha(stage/'MANIFEST.json')})
    print(json.dumps(read(out/'receipt.json')))


if __name__=='__main__':main()

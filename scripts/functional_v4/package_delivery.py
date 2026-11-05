"""Self-auditing snapshot; deduplicate identical bytes without dropping path references."""
import gzip
import json
import os
from pathlib import Path
import shutil
import tarfile
from functional_v1.common import ROOT, sha, read, write, stamp


def main():
    out=ROOT/'archives/functional-v3-v4-20261105'
    out.mkdir(parents=True,exist_ok=True)
    stage=out/'snapshot'
    assert not stage.exists(), 'Keep previous delivered snapshots immutable'
    stage.mkdir()
    files=set()
    for name in ('README.md','实验记录与结论.md','原理与复用指南.md','models/README.md','models/catalog.json'):
        files.add(ROOT/name)
    # Earlier images, parent adapters and source metadata are supplied by the verified round2 archive.
    for directory in (ROOT/'experiments/functional-v3-20261104', ROOT/'experiments/functional-v4-20261105',
                      ROOT/'data/training/functional-v3-20261104', ROOT/'data/training/functional-v4-20261105',
                      ROOT/'scripts/functional_v1', ROOT/'scripts/functional_v2', ROOT/'scripts/functional_v3', ROOT/'scripts/functional_v4',
                      ROOT/'models/from-functional-v2'):
        for p in directory.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.lock':files.add(p)
    for relative in ('experiments/functional-v1-20261104/resources.json',
                     'experiments/functional-v1-20261104/development-selection.json',
                     'experiments/functional-v2-20261104/development-selection.json',
                     'configs/functional-v3-20261104.json', 'configs/functional-v4-20261105.json',
                     'archives/functional-v2-20261104/receipt.json'):
        files.add(ROOT/relative)
    # Preserve only explicitly labelled agent browser/API uploads, never user trial data.
    for p in (ROOT/'experiments/functional-v1-20261104/demo/requests').glob('*/request.json'):
        if read(p).get('source') not in ('agent_smoke_test_not_user_feedback','agent_browser_smoke_test_not_user_feedback'):continue
        if read(p).get('model_id','').startswith(('functional-v3-', 'functional-v4-')):
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
        'not_included':['Base Qwen weights, Python environments, prior-round images and parent adapters (restore verified round2 archive first), fifth-round diagnosis/training, non-agent trial uploads.'],
        'dependency_archive':'archives/functional-v2-20261104/functional-v2-20261104.tar.gz', 'dependency_sha256':'00fd797acb73b59b8f42c1aa49f1f24aa4edb26e9a76c080755795f0f1506b11',
        'deduplication':'Identical file bytes retain every original path through tar hard links.'})
    archive=out/'functional-v3-v4-20261105.tar.gz'
    with archive.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',compresslevel=1,fileobj=raw,mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w') as tf:
        tf.add(stage,arcname='functional-v3-v4-20261105')
    # Verify every regular file and hard link by streaming from the actual archive.
    expected={r['path']:r for r in manifest};verified=[];stored={}
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf:
            relative=str(Path(member.name).relative_to('functional-v3-v4-20261105'))
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
    write(out/'receipt.json',{'status':'archive_verified','created_at':stamp(),'archive':str(archive.relative_to(ROOT)),
        'sha256':sha(archive),'bytes':archive.stat().st_size,'verified_files':len(verified),
        'unique_file_contents':len(unique),'source_logical_bytes':sum(r['bytes'] for r in manifest),
        'manifest_sha256':sha(stage/'MANIFEST.json')})
    print(json.dumps(read(out/'receipt.json')))


if __name__=='__main__':main()

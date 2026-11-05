"""Self-auditing snapshot; deduplicate identical bytes without dropping path references."""
import gzip
import json
import os
from pathlib import Path
import shutil
import tarfile
from .common import ROOT, DATA, EXP, sha, read, write, stamp


def main():
    out=ROOT/'archives/functional-v2-20261104'
    out.mkdir(parents=True,exist_ok=True)
    stage=out/'snapshot'
    assert not stage.exists(), 'Keep previous delivered snapshots immutable'
    stage.mkdir()
    files=set()
    for name in ('README.md','实验记录与结论.md','原理与复用指南.md','models/README.md','models/catalog.json'):
        files.add(ROOT/name)
    # Both parents and code are explicit dependencies; base weights/envs remain external.
    for directory in (EXP, DATA, ROOT/'scripts/functional_v1', ROOT/'scripts/functional_v2',
                      ROOT/'models/from-functional-v1', ROOT/'models/from-sft-calibrated/sft-functional/seed-01'):
        for p in directory.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.lock':files.add(p)
    files.add(ROOT/'experiments/functional-v1-20261104/resources.json')
    files.add(ROOT/'experiments/functional-v1-20261104/development-selection.json')
    files.add(ROOT/'configs/functional-v2-20261104.json')
    # Preserve only explicitly labelled agent browser/API uploads, never user trial data.
    for p in (ROOT/'experiments/functional-v1-20261104/demo/requests').glob('*/request.json'):
        if read(p).get('source') not in ('agent_smoke_test_not_user_feedback','agent_browser_smoke_test_not_user_feedback'):continue
        if read(p).get('model_id','').startswith('functional-v2-'):
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
        'not_included':['Base Qwen weights, Python environments, full first-round archive dependencies, third-round experiment, non-agent trial uploads.'],
        'deduplication':'Identical file bytes retain every original path through tar hard links.'})
    archive=out/'functional-v2-20261104.tar.gz'
    with archive.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',compresslevel=1,fileobj=raw,mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w') as tf:
        tf.add(stage,arcname='functional-v2-20261104')
    # Verify every regular file and hard link by streaming from the actual archive.
    expected={r['path']:r for r in manifest};verified=[];stored={}
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf:
            relative=str(Path(member.name).relative_to('functional-v2-20261104'))
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

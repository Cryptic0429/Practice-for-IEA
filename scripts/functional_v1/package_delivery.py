"""Self-auditing snapshot; deduplicate identical bytes without dropping path references."""
import gzip
import json
import os
from pathlib import Path
import shutil
import tarfile
from .common import ROOT, DATA, EXP, sha, read, write, stamp


def main():
    out=ROOT/'archives/functional-v1-20261104'
    out.mkdir(parents=True,exist_ok=True)
    stage=out/'snapshot'
    assert not stage.exists(), 'Keep previous delivered snapshots immutable'
    stage.mkdir()
    files=set()
    for name in ('README.md','实验记录与结论.md','原理与复用指南.md','models/README.md','models/catalog.json','configs/functional-v1-20261104.json'):
        files.add(ROOT/name)
    files.update((ROOT/'scripts').glob('*.py'))
    files.update(p for p in (ROOT/'repo/src').rglob('*.py') if p.is_file())
    files.add(ROOT/'configs/evaluation-protocol-v1.json')
    files.add(ROOT/'data/training/quality-v2-20261103/train3k.json')
    files.update(p for p in (ROOT/'models/from-sft-standard/sft-calibrated/seed-02').iterdir() if p.is_file())
    for directory in (EXP,ROOT/'scripts/functional_v1',DATA):
        for p in directory.rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.lock':continue
            relative=p.relative_to(ROOT)
            if directory==DATA and any(x in ('candidates','candidates-v1.1') for x in relative.parts):continue
            if 'requests' in relative.parts and not any(x in relative.parts for x in (
                '168537aa9bd14090b646c2c48c1427b4','911ade3c074c43e1931c1f0310c21c9e','6bd78a764b2f4809aed169e12c274d49')):continue
            files.add(p)
    selection=read(EXP/'development-selection.json')
    catalog=read(ROOT/'models/catalog.json')
    for item in catalog['models']:
        if item.get('trial_id') in selection['models']:
            files.update(p for p in (ROOT/item['canonical_path']).iterdir() if p.is_file())
    # Previously viewed regression photo is outside this experiment's normalized inputs.
    files.add(ROOT/'data/images/7QSfy_7QSfy.jpg')
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
        'not_included':['Base Qwen weights, Python environments, historical archives, pre-v1.2 candidate PNGs, non-agent trial uploads.'],
        'deduplication':'Identical file bytes retain every original path through tar hard links.'})
    archive=out/'functional-v1-20261104.tar.gz'
    with archive.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',compresslevel=1,fileobj=raw,mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w') as tf:
        tf.add(stage,arcname='functional-v1-20261104')
    # Verify every regular file and hard link by streaming from the actual archive.
    expected={r['path']:r for r in manifest};verified=[];stored={}
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf:
            relative=str(Path(member.name).relative_to('functional-v1-20261104'))
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

"""Freeze actual R19 files, retain every path, verify bytes from the built archive."""
import gzip,hashlib,json,os,shutil,tarfile
from pathlib import Path
from functional_v1.common import ROOT,read,write,stamp

NAME='functional-v19-20261107'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()

def main():
    exp=ROOT/'experiments'/NAME;out=ROOT/'archives'/NAME;stage=out/'complete-snapshot'
    assert not stage.exists() and read(exp/'development-selection.json')['status']=='failed_practical_gate'
    assert len(list((exp/'runs').glob('*/status.json')))==1
    assert all(read(p)['status']=='completed' and read(p)['step']==1002 for p in (exp/'runs').glob('*/status.json'))
    protocol=read(exp/'training-protocol.json')
    assert all(sha(ROOT/name)==h for name,h in protocol['code_hashes'].items())
    assert sha(ROOT/protocol['base_cache_path'])==protocol['base_cache_sha256']
    author_docs=read(exp/'authoritative-documents.json')
    assert all(sha(ROOT/name)==h for name,h in author_docs['sha256'].items())
    prior=ROOT/'archives/functional-v18-20261107';previous=read(prior/'complete-receipt.json')
    manifest=read(prior/'complete-snapshot/MANIFEST.json')
    assert sha(prior/'complete-snapshot/MANIFEST.json')==previous['manifest_sha256']
    dependencies=manifest['dependencies']+[dict(archive=previous['archive'],sha256=previous['sha256'])]
    assert len(dependencies)==13
    files={ROOT/name for name in author_docs['sha256']}
    for name in ('README.md','实验记录与结论.md','原理与复用指南.md','models/README.md','models/catalog.json','GITHUB_INDEX.md','docs/PROJECT_STATUS.md','docs/REPOSITORY_MAP.md','docs/ARTIFACTS.json','experiments/functional-v1-20261104/resources.json'):
        files.add(ROOT/name)
    for directory in (exp,ROOT/'data/training'/NAME,ROOT/'models/from-functional-v15/sft-photometric-bridge',ROOT/'experiments/functional-v20-20261107',ROOT/'experiments/functional-v18-20261107/report-correction',*(ROOT/'scripts'/f'functional_v{i}' for i in range(1,21))):
        for p in directory.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.lock':files.add(p)
    # Include exact called requests, not a model-name prefix or unrelated uploads.
    http=read(exp/'demo/http-integration.json');browser=read(exp/'demo/browser-integration.json')
    ids={r['request_id'] for r in http['checks']+browser['requests']}
    assert len(ids)==16
    assert len(read(exp/'demo/agent-feedback-records.json')['records'])==2
    for rid in ids:
        folder=ROOT/'experiments/functional-v1-20261104/demo/requests'/rid;r=read(folder/'request.json')
        assert r.get('source') in ('agent_round19_HTTP_test_not_user_feedback','agent_round19_browser_test_not_user_feedback')
        assert not r['training_use']
        files.update(p for p in folder.iterdir() if p.is_file())
        assert folder/'input.png' in files
        if r['status']=='executed':assert folder/'output.png' in files
        else:assert not (folder/'output.png').exists()
    for name in ('complete-receipt.json','complete-download-receipt.json','functional-v18-report-amendment-20261107.tar.gz','report-amendment-receipt.json','report-amendment-local-verification.json','report-amendment-server-verification.json'):files.add(prior/name)
    out.mkdir(parents=True,exist_ok=True);stage.mkdir();unique={};rows=[]
    for source in sorted(files):
        relative=source.relative_to(ROOT);dest=stage/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest);digest=sha(dest)
        if digest in unique:dest.unlink();os.link(unique[digest],dest)
        else:unique[digest]=dest
        rows.append(dict(path=str(relative),bytes=dest.stat().st_size,sha256=digest))
    write(stage/'MANIFEST.json',dict(at=stamp(),files=rows,dependencies=dependencies,authoritative_documents=author_docs,
        scope='Actual failed R19, both checkpoints/optimizer/scheduler/RNG, input features and fixed TRAIN audit, AI reviews, exact16 agent requests and2 feedback records. R18 report correction and amendment included. R20 is CPU-only request-span feasibility with no new model or GPU training. No qualified model, independent user feedback or final acceptance.',
        inherited_cache=dict(path=protocol['base_cache_path'],sha256=protocol['base_cache_sha256'],restore_from='archives/functional-v15-20261106/functional-v15-20261106-complete.tar.gz'),
        not_included=['Full base/author weights and environments, complete GIER source pool, prior images/checkpoints/cache restored from thirteen verified dependencies, unrelated user uploads, future work.'],
        deduplication='Every path retained; identical bytes stored through tar hard links.'))
    archive=out/(NAME+'-complete.tar.gz')
    with archive.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',compresslevel=1,fileobj=raw,mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w') as tar:
        tar.add(stage,arcname=NAME)
    expected={NAME+'/'+r['path']:r for r in rows};stored={};verified=set();hardlinks=0
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            if member.name not in expected:continue
            if member.islnk():digest,size=stored[member.linkname];hardlinks+=1
            else:
                stream=tar.extractfile(member);h=hashlib.sha256();size=0
                while chunk:=stream.read(1024*1024):h.update(chunk);size+=len(chunk)
                digest=h.hexdigest()
            row=expected[member.name];assert digest==row['sha256'] and size==row['bytes'],member.name
            stored[member.name]=(digest,size);verified.add(member.name)
    assert len(verified)==len(rows)
    receipt=dict(status='archive_verified',scope='Complete failed R19 experiment and actual trial at this point; thirteen prior dependencies; independent user feedback0; future work excluded.',
        created_at=stamp(),archive=str(archive.relative_to(ROOT)),sha256=sha(archive),bytes=archive.stat().st_size,
        verified_files=len(rows),unique_file_contents=len(unique),verified_hardlinks=hardlinks,source_logical_bytes=sum(r['bytes'] for r in rows),manifest_sha256=sha(stage/'MANIFEST.json'))
    write(out/'complete-receipt.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':main()

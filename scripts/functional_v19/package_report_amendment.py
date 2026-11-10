"""Correct the R18 narrative without rebuilding its frozen results archive."""
import argparse,hashlib,json,tarfile,io,gzip
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'archives/functional-v18-20261107'
NAME='functional-v18-report-amendment-20261107'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def verify(where):
    receipt=json.loads((FOLDER/'report-amendment-receipt.json').read_text());archive=ROOT/receipt['archive']
    assert digest(archive)==receipt['sha256'] and archive.stat().st_size==receipt['bytes']
    with tarfile.open(archive,'r:gz') as tar:
        payload=tar.extractfile(NAME+'/MANIFEST.json').read();manifest=json.loads(payload)
        assert hashlib.sha256(payload).hexdigest()==receipt['manifest_sha256']
        for row in manifest['files']:
            content=tar.extractfile(NAME+'/'+row['path']).read()
            assert len(content)==row['bytes'] and hashlib.sha256(content).hexdigest()==row['sha256']
    assert digest(ROOT/receipt['base_archive'])==receipt['base_archive_sha256']
    result=dict(at=datetime.now(timezone.utc).isoformat(),status='report_amendment_all_files_verified',where=where,
        files=len(manifest['files']),archive_sha256=receipt['sha256'],base_archive_unchanged=True)
    write(FOLDER/f'report-amendment-{where}-verification.json',result);print(json.dumps(result))

def build():
    target=FOLDER/'report-amendment-receipt.json';assert not target.exists()
    base=json.loads((FOLDER/'complete-receipt.json').read_text());assert digest(ROOT/base['archive'])==base['sha256']
    with tarfile.open(ROOT/base['archive'],'r:gz') as tar:
        payload=tar.extractfile('functional-v18-20261107/MANIFEST.json').read();original=json.loads(payload)
        assert hashlib.sha256(payload).hexdigest()==base['manifest_sha256']
        stale=tar.extractfile('functional-v18-20261107/experiments/functional-v18-20261107/REPORT.md').read().decode()
        assert '它们仍在训练' in stale
    original_files={r['path']:r for r in original['files']}
    unchanged=[f'experiments/functional-v18-20261107/{name}' for name in ('development-selection.json','epoch1-stage1.json','epoch2-stage1.json','resource-summary.json')]
    for name in unchanged:assert digest(ROOT/name)==original_files[name]['sha256'],name
    paths=[*unchanged,'experiments/functional-v18-20261107/REPORT.md',
        'experiments/functional-v18-20261107/report-correction/record.json',
        'experiments/functional-v18-20261107/report-correction/stale-before-final-correction.md',
        'experiments/functional-v19-20261107/quality-refusal-interpretation-correction.json']
    content={name:(ROOT/name).read_bytes() for name in paths}
    content['AMENDMENT.md']=('R18最终报告更正包。先恢复原完整包及12个依赖，再按相同工作区路径覆盖本包的报告；原始结果JSON与原包逐字节一致。'
        '原包的报告正文被旧同步副本覆盖而仍写训练中，本包补齐实际四检查点结果和真实数据。原模型、训练、评审与完成数均未重算。'
        '另附R19发现的星空拒绝解释：修图未完成不自动等于误拒绝；R19完整实验、图片和权重另行交付。'
        '本包是报告补充，原完整包SHA保持不变，不取代原模型与数据包。\n').encode()
    rows=[dict(path=name,bytes=len(value),sha256=hashlib.sha256(value).hexdigest()) for name,value in sorted(content.items())]
    manifest=dict(at=datetime.now(timezone.utc).isoformat(),files=rows,base_archive=base['archive'],base_archive_sha256=base['sha256'],
        unchanged_results_verified_against_base_manifest=unchanged,scope='Narrative correction and refusal interpretation only; no new training, weights, labels or functional scores.')
    manifest_bytes=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode();content['MANIFEST.json']=manifest_bytes
    archive=FOLDER/(NAME+'.tar.gz')
    with archive.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w') as tar:
        for name,value in sorted(content.items()):
            item=tarfile.TarInfo(NAME+'/'+name);item.size=len(value);item.mode=0o644;tar.addfile(item,io.BytesIO(value))
    write(target,dict(at=datetime.now(timezone.utc).isoformat(),status='report_amendment_built',archive=str(archive.relative_to(ROOT)),
        bytes=archive.stat().st_size,sha256=digest(archive),manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        base_archive=base['archive'],base_archive_sha256=base['sha256'],verified_result_files=len(unchanged),scope=manifest['scope']))
    verify('local')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify-server',action='store_true');args=parser.parse_args()
    verify('server') if args.verify_server else build()

"""Seal and verify the bounded experiment's evidence and recoverable weights."""
import gzip,hashlib,json,os,shutil,tarfile
from pathlib import Path
from .common import ROOT,EXP,read,write,stamp,protocol
from functional_v1.common import EXP as BUDGET
from functional_v22.common import protocol as p22
NAME='functional-v22-v23-20261107'

def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def check_archive(path,manifest):
 expected={NAME+'/'+r['path']:r for r in manifest['files']};stored={};hard=0
 with tarfile.open(path,'r:gz') as tar:
  for member in tar:
   if member.name not in expected:continue
   assert member.name not in stored
   if member.islnk():value=stored[member.linkname];hard+=1
   else:
    assert member.isfile();h=hashlib.sha256();size=0;f=tar.extractfile(member)
    while chunk:=f.read(1024*1024):h.update(chunk);size+=len(chunk)
    value=(h.hexdigest(),size)
   record=expected[member.name];assert value==(record['sha256'],record['bytes']),member.name;stored[member.name]=value
 assert stored.keys()==expected.keys()
 return len(stored),hard

def main():
 protocol();p22();r=read(EXP/'final-resource-summary.json');assert 23.999<=r['exact_occupied_GPU_hours']<=24 and digest(BUDGET/'resources.json')==r['ledger_sha256']
 assert read(EXP/'development-selection.json')['status']=='paired_route_not_qualified_budget_closed';r22=ROOT/'experiments/functional-v22-20261107'
 out=ROOT/'archives'/NAME;stage=out/'complete-snapshot';assert not stage.exists();stage.mkdir(parents=True)
 prior=ROOT/'archives/functional-v21-20261107';old=read(prior/'complete-receipt.json');oldmf=prior/'complete-snapshot/MANIFEST.json';assert digest(oldmf)==old['manifest_sha256']
 dependencies=read(oldmf)['dependencies']+[dict(archive=old['archive'],sha256=old['sha256'])]
 docs=('README.md','实验记录与结论.md','原理与复用指南.md','GITHUB_INDEX.md','docs/PROJECT_STATUS.md','docs/REPOSITORY_MAP.md','docs/ARTIFACTS.json','models/README.md','models/catalog.json','configs/functional-v22-20261107.json','configs/functional-v23-20261107.json')
 fixed={ROOT/n for n in docs}|{BUDGET/n for n in ('resources.json','BUDGET_STOP.json','autonomous-completion-20261107.json')}|{oldmf,prior/'complete-receipt.json',prior/'complete-download-receipt.json'}
 for name in ('web.py','worker.py','catalog.py'):fixed.add(ROOT/'scripts/functional_v1'/name)
 roots=[r22,EXP,ROOT/'data/training/functional-v22-20261107',ROOT/'data/training/functional-v23-20261107',ROOT/'scripts/functional_v22',ROOT/'scripts/functional_v23',ROOT/'scripts/functional_v1/web',ROOT/'models/from-functional-v15/quality-coverage-controller',ROOT/'models/from-functional-v15/sft-quality-coverage']
 for folder in roots:
  fixed.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.lock','.tmp') and p!=EXP/'batches.pt')
 request_ids={x['request_id'] for x in read(r22/'demo/http-integration.json')['checks']+read(r22/'demo/browser-integration.json')['requests']+read(EXP/'demo/http-integration.json')['checks']}|{'d5872254a8cf49c5903be7c39c147fd7'}
 assert len(request_ids)==23
 for rid in request_ids:
  folder=BUDGET/'demo/requests'/rid;item=read(folder/'request.json');assert item.get('source','').startswith('agent_round2') and not item['training_use']
  fixed.update(p for p in folder.iterdir() if p.is_file());assert (folder/'output.png').exists()==(item['status']=='executed')
 unique={};items=[]
 for src in sorted(fixed):
  rel=src.relative_to(ROOT);dst=stage/rel;dst.parent.mkdir(parents=True,exist_ok=True);h=digest(src)
  if h in unique:os.link(unique[h],dst)
  else:shutil.copy2(src,dst);assert digest(dst)==h;unique[h]=dst
  items.append(dict(path=str(rel),bytes=dst.stat().st_size,sha256=h))
 manifest=dict(at=stamp(),files=items,dependencies=dependencies,document_sha256={n:digest(ROOT/n) for n in docs},
  scope='R22/R23 actual admitted data, fixed weights and optimizer/RNG, new features, prediction images, original-pixel reviews, final stress prefix, actual trial requests, feedback and exact 24h-budget ledger. Sealed acceptance not opened. No claim of practical or user-accepted model.',
  omitted=[dict(path=str((EXP/'batches.pt').relative_to(ROOT)),bytes=(EXP/'batches.pt').stat().st_size,sha256=read(EXP/'training-protocol.json')['cache_sha256'],reason='11 GiB regenerable CPU tensor cache stays on training host; data and cache builder included. Used cache identity retained. Replay saved weights does not require cache.'),
   dict(reason='Full external base weights, full source dataset and prior artifacts remain dependencies; prior complete packages are referenced by SHA, not duplicated. Unrelated uploads/feedback not included.')],
  gpu_hours=r['exact_occupied_GPU_hours'],new_inference_enabled=False,known_agent_requests=23,independent_user_feedback=0)
 write(stage/'MANIFEST.json',manifest);archive=out/(NAME+'-complete.tar.gz')
 with archive.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=1,mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tar:tar.add(stage,arcname=NAME)
 n,hard=check_archive(archive,manifest)
 receipt=dict(at=stamp(),status='archive_verified',archive=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,sha256=digest(archive),manifest_sha256=digest(stage/'MANIFEST.json'),verified_files=n,verified_hardlinks=hard,unique_file_contents=len(unique),prior_archive_dependencies=len(dependencies),gpu_hours=r['exact_occupied_GPU_hours'],rebuildable_CPU_cache_not_bundled=True,acceptance_opened=False,independent_user_feedback=0)
 write(out/'complete-receipt.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()

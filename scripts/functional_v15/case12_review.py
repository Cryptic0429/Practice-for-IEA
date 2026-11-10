"""Direct review of the user's known historical regression, never a new test set."""
import datetime,hashlib,json,shutil,textwrap,os
from functional_v1.common import ROOT,read,write,sha,stamp
EXP=ROOT/'experiments/functional-v15-20261106'


def prepare():
    from PIL import Image,ImageDraw,ImageOps
    import numpy as np
    from functional_v1.curate import font
    from .editor import load_rgb
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    protocol=read(EXP/'demo/case12-protocol.json')
    assert sha(EXP/'development-selection.json')==protocol['selection_sha256']
    rows=[r for r in read(EXP/'demo/http-integration.json')['checks'] if r['case_id']==protocol['id']]
    assert len(rows)==2 and {r['language'] for r in rows}=={'zh','en'}
    out=EXP/'case12-review';assert not out.exists();out.mkdir()
    canvas=Image.new('RGB',(1800,850),'white');draw=ImageDraw.Draw(canvas)
    draw.text((12,10),'KNOWN CASE-12 REGRESSION | ANONYMOUS EXECUTED OUTPUTS',font=font(23),fill='black')
    draw.text((12,49),protocol['instructions']['en'],font=font(22),fill='black')
    assets=[];mapping={}
    requests=ROOT/'experiments/functional-v1-20261104/demo/requests'
    records=[(label,next(r for r in rows if r['language']==lang)) for label,lang in [('A','en'),('B','zh')]]
    source=requests/records[0][1]['request_id']/'input.png'
    assert np.array_equal(load_rgb(source),load_rgb(ROOT/protocol['image']))
    originals=[('INPUT',source,None)]
    for label,row in records:
        path=requests/row['request_id'];saved=read(path/'request.json')
        assert saved['identity']['adapter_sha256']==protocol['adapter_sha256'] and saved['instruction']==protocol['instructions'][row['language']]
        assert saved['source']=='agent_round15_HTTP_test_not_user_feedback' and not saved['training_use']
        assert np.array_equal(load_rgb(path/'input.png'),load_rgb(source))
        mapping[label]=dict(language=row['language'],request_id=row['request_id'],status=saved['status'])
        originals.append((label,path/'output.png' if saved['status']=='executed' else None,saved))
    for index,(label,path,saved) in enumerate(originals):
        draw.text((index*600+12,140),label,font=font(23),fill='black')
        if path is None:
            draw.text((index*600+30,420),'NO RESULT: '+saved['status'],font=font(23),fill='black');continue
        target=out/(label+'-native.png');shutil.copy2(path,target)
        assert sha(target)==sha(path)
        with Image.open(path) as im:
            rgb=im.convert('RGB');assert max(rgb.size)<=2000
            fitted=ImageOps.contain(rgb,(580,620),Image.Resampling.LANCZOS)
        canvas.paste(fitted,(index*600+10+(580-fitted.width)//2,185+(620-fitted.height)//2))
        assets.append(dict(label=label,path=str(target.relative_to(ROOT)),sha256=sha(target),source=str(path.relative_to(ROOT)),size=list(rgb.size)))
    sheet=out/'overview.jpg';canvas.save(sheet,quality=96,subsampling=0)
    write(out/'manifest.json',dict(at=stamp(),sheet=str(sheet.relative_to(ROOT)),sheet_sha256=sha(sheet),assets=assets,
        anonymous_mapping=mapping,protocol_sha256=sha(EXP/'demo/case12-protocol.json'),
        scope='One repeatedly viewed historical source. Same locked diagnostic model, literal EN and translated ZH. No independent acceptance, no new training or checkpoint tuning.'))
    print('Prepared case12 actual model outputs, no extra GPU forwards.')


def record(label,stage,completion,quality,evidence):
    assert label in ('A','B') and stage in ('overview','native')
    assert completion in 'CPFU' and quality in 'AmMU' and len(evidence)>30
    manifest=EXP/'case12-review/manifest.json';m=read(manifest)
    assert sha(ROOT/m['sheet'])==m['sheet_sha256']
    assets=[a for a in m['assets'] if a['label'] in ('INPUT',label)]
    assert all(sha(ROOT/a['path'])==a['sha256'] for a in assets)
    target=EXP/'case12-review/observations.jsonl';old=target.read_text() if target.exists() else ''
    assert (label,stage) not in {(r['label'],r['stage']) for r in map(json.loads,old.splitlines())}
    row=dict(at=stamp(),label=label,stage=stage,completion=completion,quality=quality,evidence=evidence,
        manifest_sha256=sha(manifest),assets=assets,reviewer='Codex direct anonymous visual AI review, not user approval',
        independent_acceptance=False,training_use=False,checkpoint_selection_use=False)
    target.write_text(old+json.dumps(row,ensure_ascii=False)+'\n')
    print('Recorded known case12',label,stage,completion,quality)


if __name__=='__main__':prepare()

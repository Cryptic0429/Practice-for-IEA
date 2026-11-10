"""CPU-only editor diagnosis, not training, model selection or an acceptance opening."""
import os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.curate import font
from functional_v7 import editor_soft as old
from . import candidate_editor_half_rate as half,candidate_editor_full_rate as full
EXP=ROOT/'experiments/functional-v11-20261106'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    target=EXP/'negative-exposure-probe/manifest.json';assert not target.exists()
    values=np.array([0,.1,.25,.5,.75,.9,.95,1.],np.float32)
    curves=[]
    for slider in (0.,-.0001,-10.,-30.,-60.,-100.):
        curves.append(dict(slider=slider,source=values.tolist(),current=old.bounded_gain(values,slider).tolist(),
            half_rate=half.bounded_gain(values,slider).tolist(),full_rate=full.bounded_gain(values,slider).tolist()))
    for tool in ('exposure','brightness'):
        assert np.array_equal(half.edit(np.full((8,8,3),255,np.uint8),{},'bounded-fixed'),np.full((8,8,3),255,np.uint8))
        for slider in (-100,-60,-10):
            result=half.edit(np.full((8,8,3),255,np.uint8),{tool:slider},'bounded-fixed')
            assert np.all(result<255) and np.all(result>0)
    rows=[]
    sources=read(ROOT/'experiments/functional-v10-20261106/style-candidates/manifest.json')['cases']
    originals={c['id']:c for c in read(ROOT/'experiments/functional-v10-20261106/counterfactual-candidates/manifest.json')['cases']}
    for index in (5,7,12,15):
        c=next(c for c in sources if c['id']==f'visual-risk-{index:02d}')
        source=originals[c['id']]['images']['canonical']
        assert sha(ROOT/source['path'])==source['sha256']
        rgb=old.load_rgb(ROOT/source['path'])
        images=[('INPUT',rgb)]+[(name,editor.edit(rgb,dict(exposure=-60),'bounded-fixed')) for name,editor in [('current',old),('half-rate',half),('full-rate',full)]]
        canvas=Image.new('RGB',(1200,1260),'white');draw=ImageDraw.Draw(canvas)
        draw.text((16,12),c['id']+' | training source only | darken substantially',font=font(24),fill='black')
        paths=[]
        for i,(name,pixels) in enumerate(images):
            path=EXP/'negative-exposure-probe/images'/f'{c["id"]}-{name}.png';old.save(path,pixels)
            paths.append(dict(role=name,path=str(path.relative_to(ROOT)),sha256=sha(path)))
            x,y=(i%2)*600,55+(i//2)*600;draw.text((x+12,y),name+' | exposure -60' if name!='INPUT' else name,font=font(21),fill='black')
            fitted=ImageOps.contain(Image.fromarray(pixels),(580,550),Image.Resampling.LANCZOS);canvas.paste(fitted,(x+10+(580-fitted.width)//2,y+35+(550-fitted.height)//2))
        panel=EXP/'negative-exposure-probe/panels'/f'{c["id"]}.jpg';panel.parent.mkdir(parents=True,exist_ok=True);canvas.save(panel,quality=95,subsampling=0)
        rows.append(dict(id=c['id'],domain=c['domain'],group_id=c['group_id'],images=paths,panel=str(panel.relative_to(ROOT)),panel_sha256=sha(panel)))
    write(target,dict(at=stamp(),status='CPU_candidates_not_reviewed_not_admitted',curves=curves,cases=rows,
        hypothesis='R7 negative exposure fixes the white endpoint, so large numeric negative exposure can leave a high-key photo too bright. Test negative value scaling, retaining the positive bounded branch. Half-rate matches the old midgrey infinitesimal response at zero; no model-usefulness claim.',
        parameters=dict(exposure=-60),original_training_groups_only=True,acceptance_images_read=False,GPU_hours=0,training_started=False,
        code_hashes={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}))
    print('CPU diagnosis only: white endpoint under exposure-60:',curves[-2]['current'][-1],curves[-2]['half_rate'][-1],curves[-2]['full_rate'][-1])

if __name__=='__main__':main()

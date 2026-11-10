"""Native-pixel exports for three maximum-strength training-pool probes."""
import os
from PIL import Image
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb,save


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    exp=ROOT/'experiments/functional-v15-20261106';out=exp/'shadow-tool-probes/native-pixels';assert not (out/'manifest.json').exists();rows=[]
    for item in read(exp/'shadow-tool-probes/manifest.json')['items']:
        if item['parameters']['shadows']!=90:continue
        images=[]
        for role,path in [('INPUT',item['input']),('OLD',item['old_output']),('PROTOTYPE',item['new_output'])]:
            source=ROOT/path;rgb=load_rgb(source);assert max(rgb.shape[:2])<=1200
            target=out/f'{item["source_id"]}-{role}.png';save(target,rgb)
            assert __import__('numpy').array_equal(load_rgb(target),rgb)
            images.append(dict(role=role,path=str(target.relative_to(ROOT)),sha256=sha(target),size=list(rgb.shape[1::-1]),source=path,source_sha256=sha(source)))
        rows.append(dict(source_id=item['source_id'],parameters=item['parameters'],images=images))
    write(out/'manifest.json',dict(at=stamp(),items=rows,transform='Whole native pixels, no resampling or adjustment; training-pool source/tool comparison, no model prediction or admitted label.',GPU_hours=0))
    print('9 native-pixel training probe images exported.')


if __name__=='__main__':main()

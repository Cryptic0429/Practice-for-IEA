"""Additional real executed attempts after the original-pixel rock review failed."""
import os
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb,save
from .prepare_labels import EXP,panel
from .outcomes import execute


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    folder=EXP/'quality-revalidation/repairs/wave-natural-colour';assert not (folder/'manifest.json').exists()
    c=next(c for c in read(EXP/'candidate-cases.json') if c['id']=='native-train-n190');rgb=load_rgb(ROOT/c['image'])
    rows=[]
    for i,p in enumerate([dict(exposure=-45,brightness=45,contrast=60,highlights=-20,shadows=20,saturation=5),
                          dict(exposure=-50,brightness=50,contrast=55,highlights=-15,shadows=35,saturation=5)],6):
        result=execute(rgb,dict(action='edit',parameters=p));file=folder/f'alt{i}.png';save(file,result['result'])
        rows.append(dict(id=f'native-train-n190-alt{i}',case_id=c['id'],parameters=p,input=dict(path=c['image'],sha256=c['input_sha256']),output=dict(path=str(file.relative_to(ROOT)),sha256=sha(file)),status='CPU_executed_not_viewed_not_admitted'))
    sheet=panel(c['id'],c['instructions']['en'],[('INPUT',ROOT/c['image']),*[(r['id'],ROOT/r['output']['path']) for r in rows]],folder/'panel.jpg')
    write(folder/'manifest.json',dict(at=stamp(),case_id=c['id'],instructions=c['instructions'],candidates=rows,panel=sheet,
        reason='Compare lower extra saturation with the same strong tonal-contrast compensation. Earlier candidates and their original-pixel evidence remain separate; no new candidate is admitted before full review.',GPU_hours=0,training_started=False))
    print('Two new wave/rock-detail attempts executed; no acceptance or training decision.')

if __name__=='__main__':main()

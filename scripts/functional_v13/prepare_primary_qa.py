"""Native-pixel input/selected-result QA copies and explicitly bounded crops."""
import argparse,json
from PIL import Image
from functional_v1.common import ROOT,read,write,sha,stamp
from .source_candidates import EXP


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True)
    ap.add_argument('--candidates',default='primary-candidates');ap.add_argument('--observations',default='primary-observations.jsonl')
    ap.add_argument('--prior',default='primary-qa-observations.jsonl');args=ap.parse_args()
    assert args.name.replace('-','').isalnum()
    dest=EXP/'primary-qa'/args.name;assert not (dest/'manifest.json').exists()
    bank={r['id']:r for r in read(EXP/args.candidates/'manifest.json')['items']}
    notes=list(map(json.loads,(EXP/args.observations).read_text().splitlines()))
    prior=EXP/args.prior;done={r['id'] for r in map(json.loads,prior.read_text().splitlines())} if prior.exists() else set()
    boxes={
        'train-native-117':{'skin-and-flowers':[100,150,850,1150]},
        'train-native-212':{'male-face':[800,350,2000,1550],'female-face-hair':[2200,400,3400,1600]},
        'train-native-175':{'face-shirt':[100,100,800,1000]},
        'train-native-040':{'central-skin-cloth':[1500,600,2700,1800],'left-skin-cloth':[400,500,1600,1700]},
        'train-native-135':{'central-face-noise':[400,200,1600,1400]},
        'train-native-074':{'baby-bedding':[350,100,1550,1300]},
        'train-extension-211':{'windows-sky':[1000,250,2200,1450]},
        'train-native-114':{'letters-foreground':[1100,1000,2300,2200],'letters-right':[2200,1000,3400,2200]},
        'train-extension-043':{'sun-water':[1360,150,2560,1350],'beach-sky':[100,150,1300,1350]},
        'train-extension-521':{'sky-points':[650,300,1850,1500],'skyline':[650,2064,1850,3264]},
        'train-extension-552':{'star-trails':[900,50,2100,1250],'branches':[100,456,1300,1656]},
        'train-extension-446':{'rigging-sky':[300,100,1500,1300]},
        'train-extension-422':{'central-foliage':[1000,600,2200,1800]},
    }
    items=[]
    for n in notes:
        if n['quality']!='U' or n['id'] in done or n['chosen_candidate'] is None:continue
        item=bank[n['id']];result=next(r for r in item['results'] if r['candidate']==n['chosen_candidate']);assert result['image']
        images=[]
        for role,relative,digest in [('INPUT',item['input_image'],item['input_sha256']),('SELECTED',result['image'],result['image_sha256'])]:
            path=ROOT/relative;assert sha(path)==digest
            with Image.open(path) as im:
                im=im.convert('RGB');w,h=im.size
                regions=boxes.get(item.get('source_id',n['id']),{'whole-frame':[0,0,w,h]})
                for label,box in regions.items():
                    assert 0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h
                    target=dest/n['id']/f'{label}-{role}.png';target.parent.mkdir(parents=True,exist_ok=True)
                    crop=im.crop(box);crop.save(target)
                    images.append(dict(role=role,region=label,path=str(target.relative_to(ROOT)),sha256=sha(target),
                        source=relative,source_sha256=digest,source_size=[w,h],box=box,size=list(crop.size)))
        items.append(dict(id=n['id'],chosen_candidate=n['chosen_candidate'],output_sha256=result['image_sha256'],
            input_sha256=item['input_sha256'],images=images))
    write(dest/'manifest.json',dict(at=stamp(),items=items,
        transformations='Integer-coordinate crop or whole original-size RGB copy only, no resizing, colour adjustment, sharpening or synthesis. Same crop coordinates for input/result. Whole-frame context stays on original review sheet.',GPU_hours=0))
    print('Native-pixel QA items',len(items))


if __name__=='__main__':main()

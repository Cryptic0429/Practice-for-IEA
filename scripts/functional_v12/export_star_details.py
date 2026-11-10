"""Export matching native-pixel star/foreground QA regions, without resizing."""
import json
from pathlib import Path
from PIL import Image
from functional_v1.common import ROOT, sha, write
from .runtime import EXP


def main():
    full = EXP / 'review/native-development/prompt-control/full-resolution'
    item = next(r for r in json.loads((full / 'manifest.json').read_text())['items']
                if r['case_id'] == 'native-dev-032')
    dest = full.parent / 'star-details'
    assert not (dest / 'manifest.json').exists()
    dest.mkdir(parents=True, exist_ok=True)
    regions = {'sky': [1900, 500, 3100, 1500], 'foreground': [600, 3100, 1800, 3744]}
    images = []
    for region, box in regions.items():
        for source in item['images']:
            path = ROOT / source['path']
            assert sha(path) == source['sha256']
            with Image.open(path) as im:
                target = dest / f'{region}-{source["role"]}.png'
                crop = im.convert('RGB').crop(box)
                crop.save(target)
                images.append(dict(role=source['role'], region=region, source=source,
                    box=box, path=str(target.relative_to(ROOT)), sha256=sha(target), size=list(crop.size)))
    write(dest / 'manifest.json', dict(item_id=item['item_id'], case_id=item['case_id'],
        input_sha256=item['input_sha256'], output_sha256=item['output_sha256'], images=images,
        transformations='Integer crop only; same coordinates, no resizing or colour adjustment.'))
    print('Exported', len(images), 'native-pixel regions')


if __name__ == '__main__':
    main()

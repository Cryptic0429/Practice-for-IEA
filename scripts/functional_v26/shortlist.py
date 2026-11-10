"""Draw new train-pool source previews using the existing scan, without creating labels."""
import re

from PIL import Image, ImageOps

from functional_v1.common import ROOT, read, sha, write
from functional_v24.probe import draw_cell

EXP = ROOT/'experiments/functional-v26-20261107'
R24 = ROOT/'experiments/functional-v24-20261107'
SCAN = R24/'scan.json'
OUTDOOR = re.compile(
    r'\b(sky|landscape|scenery|scenic|mountain[s]?|sunset|sunrise|cloud[s]?|waterfall|forest|cityscape|'
    r'beach|river|woodland|milky way|night sky|stars|seascape|coast|waterfront|moon|snow|lake|trees|wood[s]?)\b',
    re.I,
)


def outdoor_previews(rows):
    annotations = ROOT/'repo/datasets/GIER/GIER.json'
    data = read(annotations)
    selected = []
    for row in rows:
        text = ' '.join(' '.join(str(data[i][k]) for k in ('request','expert_summary','amateur_summary'))
                        for i in row['annotation_indices'])
        if OUTDOOR.search(text) and min(row['size']) >= 350 and row['dark_share'] > .07:
            selected.append(row)
    for start in range(0,len(selected),14):
        sheet = Image.new('RGB',(1050,450),'white')
        for i,row in enumerate(selected[start:start+14]):
            with Image.open(ROOT/row['source']) as image:
                draw_cell(sheet,ImageOps.exif_transpose(image).convert('RGB'),
                          i%7*150,i//7*225,(150,225),row['id'])
        sheet.save(EXP/f'outdoor-{start//14+1}.jpg',quality=92)
    write(EXP/'outdoor-shortlist.json',dict(scan_sha256=sha(SCAN),annotations_sha256=sha(annotations),
        expression=OUTDOOR.pattern,minimum_short_side=350,minimum_dark_share=.07,
        rows=[r['id'] for r in selected]))


def main():
    scan = read(SCAN)
    seen = set(scan['shortlist'])
    seen_groups = {r['group_id'] for r in scan['rows'] if r['id'] in seen}
    selected = []
    for cue in ('portrait', 'landscape', 'uncued'):
        for dark_min, dark_max in ((.06,.21),(.21,.58),(.58,1.0)):
            rows = [r for r in scan['rows'] if r['group_id'] not in seen_groups
                    and (cue in r['cues'] if cue != 'uncued' else not r['cues'])
                    and dark_min <= r['dark_share'] < dark_max and r['median_luma'] < 160
                    and max(r['size']) >= 480 and min(r['size']) >= 300]
            rows.sort(key=lambda r:r['dark_chroma_residual'])
            # Across a range of roughness, avoiding selection of only heavily damaged images.
            if len(rows) > 6:
                rows = [rows[round((len(rows)-1)*q)] for q in (.15,.30,.45,.60,.75,.90)]
            for r in rows:
                if r['group_id'] not in seen_groups:
                    selected.append(r)
                    seen_groups.add(r['group_id'])
    EXP.mkdir(parents=True, exist_ok=True)
    for start in range(0,len(selected),15):
        sheet = Image.new('RGB',(1100,618),'white')
        for i,row in enumerate(selected[start:start+15]):
            title = f"{row['id']} / {','.join(row['cues']) or '?'} / {row['dark_share']:.2f}"
            with Image.open(ROOT/row['source']) as image:
                draw_cell(sheet,ImageOps.exif_transpose(image).convert('RGB'),i%5*220,i//5*206,(220,206),title)
        sheet.save(EXP/f'sources-{start//15+1}.jpg',quality=91)
    write(EXP/'shortlist.json',dict(scan_sha256=sha(SCAN),rows=[r['id'] for r in selected]))
    outdoor_previews(scan['rows'])
    print('Source previews:',len(selected))


if __name__ == '__main__':
    main()

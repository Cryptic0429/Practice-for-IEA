"""Prepare anonymous input/before/after sheets and native copies for editor control."""
import json, os, textwrap
from PIL import Image, ImageDraw, ImageOps
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.curate import font
from .native_labels import EXP


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    dest = EXP / 'editor-only-review'
    assert not dest.exists(); dest.mkdir()
    rows = []
    for split in ('native-development', 'development'):
        predictions = EXP / 'editor-only' / split / 'predictions.jsonl'
        bank = [json.loads(line) for line in predictions.read_text().splitlines()]
        grouped = {}
        for r in bank:
            if r['pixels_changed_by_editor']:
                grouped.setdefault((r['case_id'], r['output_sha256']), []).append(r)
        for (cid, digest), values in grouped.items():
            r = values[0]; old = r['original_prediction']
            key = cid + '-' + digest[:12]; folder = dest / key; folder.mkdir()
            paths = {'INPUT':r['input_image'], 'OLD':old['output_image'], 'NEW':r['output_image']}
            expected = {'INPUT':r['input_sha256'], 'OLD':old['output_sha256'], 'NEW':digest}
            assert all(sha(ROOT / paths[k]) == expected[k] for k in paths)
            # Alternation is a reproducible anonymous display order, not a selection rule.
            order = ['OLD','NEW'] if int(digest[:2],16) % 2 else ['NEW','OLD']
            canvas = Image.new('RGB',(1800,820),'white'); draw=ImageDraw.Draw(canvas)
            draw.text((12,10),cid+' | ANONYMOUS IMAGE COMPARISON',font=font(22),fill='black')
            for j, line in enumerate(textwrap.wrap(old['instruction'],130)):
                draw.text((12,45+25*j),line,font=font(20),fill='black')
            native=[]
            for n,(label,role) in enumerate([('INPUT','INPUT'),('A',order[0]),('B',order[1])]):
                with Image.open(ROOT/paths[role]) as im:
                    rgb=im.convert('RGB'); assert max(rgb.size)<=2000
                    fitted=ImageOps.contain(rgb,(580,600),Image.Resampling.LANCZOS)
                    path=folder/(label+'-native.png'); rgb.save(path,compress_level=1)
                    with Image.open(path) as check: assert check.tobytes()==rgb.tobytes()
                draw.text((n*600+12,150),label,font=font(22),fill='black')
                canvas.paste(fitted,(n*600+10+(580-fitted.width)//2,190+(600-fitted.height)//2))
                native.append(dict(label=label,path=str(path.relative_to(ROOT)),sha256=sha(path),
                                   source=paths[role],source_sha256=expected[role],size=list(rgb.size)))
            sheet=folder/'review.jpg';canvas.save(sheet,quality=96,subsampling=0)
            rows.append(dict(key=key,case_id=cid,split=split,language_rows=[r['language'] for r in values],
                instructions={r['language']:r['instruction'] for r in values},source_predictions_sha256=sha(predictions),
                sheet=str(sheet.relative_to(ROOT)),sheet_sha256=sha(sheet),native=native,
                anonymous_mapping=dict(A=order[0],B=order[1]),model_identity_hidden_in_sheet=True,
                old_output_sha256=old['output_sha256'],new_output_sha256=digest))
    write(dest/'manifest.json',dict(at=stamp(),items=rows,unique_items=len(rows),model_forwards=0,GPU_hours=0,
        policy='Anonymous tool comparison only. No learning or independent benchmark claim; native images unscaled and pixel-verified.'))
    print('Prepared',len(rows),'unique changed editor-control comparisons')


if __name__ == '__main__':main()

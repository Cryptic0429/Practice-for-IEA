"""Preserve initial sheets; add readable English headers from unchanged case files."""
import os, textwrap
from PIL import Image, ImageDraw
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.curate import font
from .native_labels import EXP


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    folder=EXP/'editor-only-review'; source=folder/'manifest.json'
    dest=folder/'manifest-readable.json'; assert not dest.exists()
    manifest=read(source); protocol=read(EXP/'editor-only-protocol.json')
    cases={split:{c['id']:c for c in read(ROOT/info['cases']['path'])}
           for split,info in protocol['sources'].items()}
    for item in manifest['items']:
        old=ROOT/item['sheet']; assert sha(old)==item['sheet_sha256']
        with Image.open(old) as im: canvas=im.convert('RGB')
        draw=ImageDraw.Draw(canvas);draw.rectangle((0,0,canvas.width,145),fill='white')
        draw.text((12,10),item['case_id']+' | ANONYMOUS IMAGE COMPARISON',font=font(22),fill='black')
        instructions=cases[item['split']][item['case_id']]['instructions']
        for j,line in enumerate(textwrap.wrap(instructions['en'],130)):
            draw.text((12,45+25*j),line,font=font(20),fill='black')
        path=old.with_name('review-en.jpg');assert not path.exists();canvas.save(path,quality=96,subsampling=0)
        item.update(initial_sheet=item['sheet'],initial_sheet_sha256=item['sheet_sha256'],
            sheet=str(path.relative_to(ROOT)),sheet_sha256=sha(path),displayed_request=instructions['en'],
            unchanged_case_requests=instructions)
    manifest.update(header_repair_at=stamp(),initial_manifest_sha256=sha(source),
        reason='Initial DejaVu font lacks Chinese glyphs. Preserve those initial sheets; display the exact original English case request, with both original requests retained in metadata. No parameter/pixel-result/grade changes.')
    write(dest,manifest); print('Added readable original-English headers to',len(manifest['items']),'sheets')


if __name__=='__main__':main()

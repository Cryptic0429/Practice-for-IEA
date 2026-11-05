"""Visual-review transport; kept separate from the frozen editing inference path."""
import json
from PIL import Image, ImageDraw, ImageOps
from .common import image_path
from .runtime import image_input, generate, REVIEW_SYSTEM


def parse_review(response):
    text = response.strip()
    # Review annotations may be fenced or introduced by prose. Only a genuine
    # JSON object with the complete rubric is accepted; missing ratings are never filled in.
    start = text.find('{')
    if start < 0:
        raise ValueError('Reviewer returned no JSON object: ' + repr(text[:800]))
    try:
        result, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError as error:
        raise ValueError('Unparseable reviewer response: ' + repr(text[:800])) from error
    allowed = {'completion': {'completed','partial','failed','uncertain'},
               'quality': {'acceptable','minor_issue','major_issue','uncertain'},
               'strength': {'appropriate','insufficient','excessive','uncertain'}}
    for field, values in allowed.items():
        if result.get(field) not in values:
            raise ValueError('Invalid visual-review field '+field+': '+repr(text[:800]))
    if not isinstance(result.get('evidence'), str) or not result['evidence'].strip():
        raise ValueError('Visible review evidence is missing: '+repr(text[:800]))
    result['raw_response'] = response
    return result


def judge(model, p, case, output):
    request = (case['instructions']['en']+'\nChinese equivalent: '+case['instructions']['zh']+
               '\nRequired change: '+case['expected_change']+'\nPreserve: '+case['must_preserve']+
               '\nReturn the four rubric fields as a JSON object. No editing parameters are requested.')
    # The same aligned sheet makes subtle relative edits easier to compare than
    # two separate image-token sequences. Full frames, no colour edits or crops.
    sheet = Image.new('RGB', (1240, 680), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (label, path) in enumerate((('BEFORE', image_path(case)), ('AFTER', output))):
        draw.text((20+620*i, 12), label, fill='black')
        with Image.open(path) as im:
            fitted = ImageOps.contain(im.convert('RGB'), (600, 620))
        sheet.paste(fitted, (10+620*i+(600-fitted.width)//2, 45+(620-fitted.height)//2))
    system = REVIEW_SYSTEM.replace('The first image is BEFORE and the second is AFTER.',
             'One aligned contact sheet shows BEFORE on the left and AFTER on the right.')
    system += (' Judge the requested RELATIVE change, not an absolute ideal brightness. '
               'Only new or worsened damage counts as an editing side effect; '
               'do not blame the edit for defects already visible in BEFORE. '
               'A deliberately modest change can fully satisfy a modest request.')
    msg = [{'role': 'system', 'content': system},
           {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': request}]}]
    old_max = p.image_processor.max_pixels
    p.image_processor.max_pixels = 524288
    try:
        batch = p(text=[p.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)],
                  images=[sheet], return_tensors='pt', add_special_tokens=False)
    finally:
        p.image_processor.max_pixels = old_max
    response = generate(model, p, batch, max_tokens=256)
    return parse_review(response)

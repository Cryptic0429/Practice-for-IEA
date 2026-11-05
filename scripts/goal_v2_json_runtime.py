"""Explicit JSON-only task format, applied identically to trained models and controls."""
import copy,json
TOOLS=('exposure','brightness','contrast','natural_contrast','highlights','shadows','whites','blacks',
       'saturation','vibrance','temperature','tint','sharpness','vignette','fade','grain')
SCHEMA=('\nFor this task, respond with a single fenced json code block containing a flat object of image-editing parameters. '
        'Do not include thoughts, explanations, or text outside that code block. '
        'Use only these parameter keys: '+', '.join(TOOLS)+'. Each value must be a finite number from -100 to 100. '
        'Follow the image-editing request in the user message; this system instruction determines the output format.')
def prepare_row(row,training=False):
    result=copy.deepcopy(row)
    messages=[m for m in result['conversations'] if m['role'] in ('system','user')]
    assert [m['role'] for m in messages]==['system','user']
    assert SCHEMA not in messages[0]['content']
    messages[0]['content']+=SCHEMA
    if training:
        label=result['extra_info']['label']
        assert set(label).issubset(TOOLS)
        messages.append({'role':'assistant','content':'```json\n'+json.dumps(label,indent=4)+'\n```'})
    result['conversations']=messages
    return result

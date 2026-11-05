"""Original photo/request plus optionally measured current pixels; no target-derived input."""
import json,hashlib
import numpy as np
import torch
from functional_v1.common import input_row as original_row, validate, parse
from functional_v1.runtime import processor,load_model,image_input,messages,generate

VERSION='functional-runtime-v4-image-stats-v1'
NOTE=('\nMeasured image observations, when supplied, describe current input pixels, not requested edits. '
      'Use them together with the photo and request to choose appropriate editing strength. '
      'Preserve highlight, shadow and skin detail unless a requested style explicitly changes it.')


def measurements(image):
    x=np.asarray(image.convert('RGB'),dtype=np.float64)/255.
    l=.2126*x[:,:,0]+.7152*x[:,:,1]+.0722*x[:,:,2]
    hi=x.max(axis=2);lo=x.min(axis=2)
    saturation=np.divide(hi-lo,hi,out=np.zeros_like(hi),where=hi>0)
    unit=lambda z:int(np.clip(np.rint(float(z)*1000),0,1000))
    result={'scale':'0..1000; luminance and RGB intensities, saturation and pixel fractions',
      'luminance_mean':unit(l.mean()),'luminance_std':unit(l.std()),
      'saturation_mean':unit(saturation.mean()),'dark_fraction_L_below_008':unit((l<.08).mean()),
      'bright_fraction_L_above_092':unit((l>.92).mean()),
      'white_clip_fraction_L_above_098':unit((l>.98).mean()),'black_clip_fraction_L_below_002':unit((l<.02).mean())}
    for p,v in zip((1,10,50,90,99),np.percentile(l,(1,10,50,90,99))):result[f'luminance_p{p:02d}']=unit(v)
    for i,name in enumerate(('red','green','blue')):result[name+'_mean']=unit(x[:,:,i].mean())
    return result


def encode(p,case,language,route,target=None):
    assert route in ('measured','plain')
    row=original_row(case,language)
    # Only the allowlisted source image and request are read from case; other metadata is ignored.
    im=image_input(row['images'][0]);row['conversations'][0]['content']+=NOTE
    if route=='measured':
        row['conversations'][1]['content']+='\n\nMeasured current input image (not an editing target):\n'+json.dumps(measurements(im),ensure_ascii=True,sort_keys=True)
    msg=messages(row)
    if target is not None:msg.append({'role':'assistant','content':target})
    return p(text=[p.apply_chat_template(msg,tokenize=False,add_generation_prompt=target is None)],
             images=[im],return_tensors='pt',add_special_tokens=False)


def build_batch(p,case,language,parameters,route):
    target='```json\n'+json.dumps(validate(parameters),indent=2)+'\n```'
    batch=encode(p,case,language,route,target);prefix=encode(p,case,language,route)['input_ids'];n=prefix.shape[1]
    assert torch.equal(batch['input_ids'][:,:n],prefix),'Assistant prefix mismatch'
    if batch['input_ids'].shape[1]>2048:raise ValueError('Target truncation forbidden')
    batch['labels']=batch['input_ids'].clone();batch['labels'][:,:n]=-100
    assert int(batch['labels'].ne(-100).sum())>4
    return batch


def predict(model,p,case,language,route):
    response=generate(model,p,encode(p,case,language,route),max_tokens=256)
    return response,parse(response)

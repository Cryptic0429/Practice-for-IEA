"""Isolated negative exposure shoulder for R30 candidate comparisons."""
import numpy as np

from functional_v15.outcomes import execute
from functional_v30.constants import RECIPES

LUMA = np.array([.2126,.7152,.0722],np.float32)


def darken_with_shoulder(rgb,value):
    if not -100<=value<=0:
        raise ValueError('Expected a nonpositive exposure value')
    img = rgb.astype(np.float32)/255
    lum = img @ LUMA
    t = np.clip((lum-.68)/.32,0,1)
    shoulder = t*t*(3-2*t)
    factor = 2**(value/100)
    factor = factor+(1-factor)*shoulder
    return np.rint(img*factor[...,None]*255).astype(np.uint8)


def apply(rgb,recipe):
    mode,value = RECIPES[recipe]
    if mode=='linear':
        return execute(rgb,dict(action='edit',parameters={'exposure':value}))['result']
    return darken_with_shoulder(rgb,value)

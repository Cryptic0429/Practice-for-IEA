"""Fixed input-RGB features; no target, request, case identity or corpus fitting."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps

FEATURE_NAMES=('luminance_mean','luminance_std','luminance_p05','luminance_p25','luminance_p50','luminance_p75','luminance_p95',
               'dark_fraction','bright_fraction','red_mean','green_mean','blue_mean','red_minus_green','blue_minus_green','saturation_mean','saturation_std')

def features(path):
    with Image.open(Path(path)) as source:
        image=ImageOps.exif_transpose(source).convert('RGB')
        image.thumbnail((512,512),Image.Resampling.LANCZOS)
        rgb=np.asarray(image,dtype=np.float32)/255.
    assert rgb.ndim==3 and rgb.shape[2]==3 and np.isfinite(rgb).all()
    lum=rgb@np.array([.2126,.7152,.0722],dtype=np.float32)
    maxima=rgb.max(-1);sat=(maxima-rgb.min(-1))/np.maximum(maxima,1/255)
    means=rgb.mean((0,1),dtype=np.float64)
    values=[lum.mean(),lum.std(),*np.quantile(lum,[.05,.25,.5,.75,.95]),(lum<.1).mean(),(lum>.9).mean(),*means,
            means[0]-means[1],means[2]-means[1],sat.mean(),sat.std()]
    assert len(values)==16
    # Fixed centering, never trained/estimated from validation or labels.
    center=np.array([.5,0,.5,.5,.5,.5,.5,0,0,.5,.5,.5,0,0,.5,0],dtype=np.float32)
    result=np.asarray(values,dtype=np.float32)-center
    assert result.shape==(16,) and np.isfinite(result).all()
    return result

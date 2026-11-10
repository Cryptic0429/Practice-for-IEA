"""Fixed current-RGB residual observations, not a noise oracle or image editor."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageFilter

NAMES=('lum_residual_mean','lum_residual_std','lum_residual_p95',
       'chroma_residual_mean','chroma_residual_p95','dark_lum_residual_mean',
       'dark_chroma_residual_mean','dark_relative_lum_residual',
       'dark_relative_chroma_residual','bright_lum_residual_mean',
       'median_lum_gradient_mean','residual_to_gradient_ratio')

def features(path):
    with Image.open(Path(path)) as source:
        image=ImageOps.exif_transpose(source).convert('RGB')
        image.thumbnail((512,512),Image.Resampling.LANCZOS)
        rgb=np.asarray(image,dtype=np.float32)/255.
        median=np.asarray(image.filter(ImageFilter.MedianFilter(3)),dtype=np.float32)/255.
    weights=np.array([.2126,.7152,.0722],dtype=np.float32)
    lum=rgb@weights;medlum=median@weights
    delta=rgb-median
    residual=np.abs(lum-medlum)
    chroma=(np.abs(delta[...,0]-delta[...,1])+np.abs(delta[...,2]-delta[...,1]))/2
    dark=lum<.15;bright=lum>.8
    def masked(a,mask):return float(a[mask].mean()) if mask.any() else 0.
    dl=masked(residual,dark);dc=masked(chroma,dark);dark_mean=masked(lum,dark)
    gradients=[]
    if medlum.shape[0]>1:gradients.append(float(np.abs(np.diff(medlum,axis=0)).mean()))
    if medlum.shape[1]>1:gradients.append(float(np.abs(np.diff(medlum,axis=1)).mean()))
    edge=float(np.mean(gradients)) if gradients else 0.
    values=[residual.mean(),residual.std(),np.quantile(residual,.95),chroma.mean(),np.quantile(chroma,.95),
            dl,dc,dl/(dark_mean+.01),dc/(dark_mean+.01),masked(residual,bright),edge,float(residual.mean())/(edge+.005)]
    result=np.asarray(values,dtype=np.float32)
    assert result.shape==(12,) and np.isfinite(result).all()
    return result

"""Offline CPU editor for the R31 overall-darkening diagnostic."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from functional_v1.common import read,sha
from functional_v15.editor import load_rgb,save
from functional_v30.constants import RECIPES
from functional_v30.editor import apply
from functional_v30.select import features
from .face import FaceRegions,measures
from .render import EXP,PLAN
from .router import Router


class Editor:
    def __init__(self):
        fit=read(EXP/'fit.json')
        if fit['protocol_sha256']!=sha(PLAN):
            raise ValueError('Fitted selector does not match the runtime protocol')
        self.params=fit['models']['region']
        self.faces=FaceRegions()
        if fit['local_detector_hashes']!={p.name:sha(p) for p in self.faces.paths}:
            raise ValueError('Local face detector weights changed')

    def edit(self,path,instruction,route):
        if route['guarded']!='global_down':
            return dict(status='declined',reason='route_not_confirmed'),None
        source=load_rgb(path)
        candidates={name:apply(source,name) for name in RECIPES}
        global_features,_=features(source,candidates,instruction)
        face,located=measures(self.faces,source,candidates,{'zh':instruction,'en':instruction})
        if located['face_required'] and not located['boxes']:
            return dict(status='declined',reason='face_location_unresolved'),None
        x=np.c_[global_features,face]
        p=expit(((x-self.params['mean'])/self.params['scale']) @
                np.asarray(self.params['coefficient']).T+self.params['intercept']).ravel()
        i=int(p.argmax())
        if p[i]<self.params['threshold']:
            return dict(status='declined',reason='quality_unreliable',confidence=float(p[i])),None
        name=list(RECIPES)[i]
        return dict(status='edited',recipe=name,confidence=float(p[i])),candidates[name]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--image',type=Path,required=True)
    ap.add_argument('--instruction',required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output file')
    route=Router().classify([args.instruction])[0]
    result,pixels=Editor().edit(args.image,args.instruction,route)
    if pixels is not None:
        save(args.output,pixels)
        result['output']=str(args.output)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()

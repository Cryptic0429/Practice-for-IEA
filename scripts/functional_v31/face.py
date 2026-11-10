"""Locate faces on source pixels, then measure the same locations in each candidate."""
from pathlib import Path
import re

import cv2
import numpy as np

from functional_v30.editor import LUMA

FACE = re.compile(r'脸|面部|五官|表情|眼睛|眼部|\b(?:faces?|facial|expressions?|eyes?)\b',re.I)
FACE_NAMES = ('face_required','location_unresolved','source_face_median','result_face_median',
              'result_face_p20','result_face_drop')


class FaceRegions:
    def __init__(self):
        p=Path(cv2.data.haarcascades)
        self.paths=[p/'haarcascade_frontalface_default.xml',p/'haarcascade_profileface.xml']
        self.frontal,self.profile=(cv2.CascadeClassifier(str(x)) for x in self.paths)
        if self.frontal.empty() or self.profile.empty():
            raise ValueError('Local OpenCV face classifiers are missing')
        self.clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))

    def detect(self,source):
        h,w=source.shape[:2]
        ratio=min(1,640/max(h,w))
        sw,sh=max(1,round(w*ratio)),max(1,round(h*ratio))
        gray=cv2.cvtColor(cv2.resize(source,(sw,sh),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY)
        boxes=[]
        for view in (cv2.equalizeHist(gray),self.clahe.apply(gray)):
            for method,flip in ((self.frontal,False),(self.profile,False),(self.profile,True)):
                for x,y,a,b in method.detectMultiScale(cv2.flip(view,1) if flip else view,
                                                      scaleFactor=1.1,minNeighbors=5,minSize=(22,22)):
                    if flip:
                        x=sw-x-a
                    box=(int(x),int(y),int(x+a),int(y+b))
                    if not any(overlap(box,other)>.35 for other in boxes):
                        boxes.append(box)
        return [(round(x*w/sw),round(y*h/sh),round(a*w/sw),round(b*h/sh)) for x,y,a,b in boxes]


def overlap(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return area/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


def interior(rgb,box):
    x,y,a,b=box
    dx,dy=round((a-x)*.20),round((b-y)*.19)
    return rgb[y+dy:b-dy,x+dx:a-dx].astype(np.float32) @ LUMA/255


def measures(detector,source,outputs,requests):
    required=bool(FACE.search(requests['zh']))
    if required!=bool(FACE.search(requests['en'])):
        raise ValueError('Literal face-preservation cues differ by language')
    boxes=detector.detect(source) if required else []
    rows=[]
    for out in outputs.values():
        if not boxes:
            rows.append([int(required),int(required),0,0,0,0])
            continue
        before=[float(np.median(interior(source,b))) for b in boxes]
        after=[interior(out,b) for b in boxes]
        med=[float(np.median(r)) for r in after]
        rows.append([1,0,min(before),min(med),min(float(np.quantile(r,.2)) for r in after),
                     max(a-b for a,b in zip(before,med))])
    return np.asarray(rows),dict(face_required=required,boxes=boxes)

"""Online path calls frozen parent on literal inputs, then the learned head."""
from .common import ROOT,EXP,PARENT,PARENT_SHA,POLICIES,protocol,sha
from .head import load_head,predict_cached
from .texture import features as texture_features
from functional_v15.runtime import processor,verify_origin
from functional_v15.outcomes import *

def load_model(route,adapter=None):
    from functional_v15.runtime import load_model as load_backbone
    assert route in POLICIES and adapter is not None
    p=protocol();head,item=load_head(adapter)
    assert item['parent_sha256']==PARENT_SHA and item['protocol_sha256']==sha(EXP/'training-protocol.json')
    return load_backbone('language_merger',PARENT),head,item,route

def predict(model,p,case,language):
    from functional_v15.runtime import predict as predict_parent
    from functional_v20.features import extract
    import torch
    backbone,head,item,policy=model
    parent_response,parent=predict_parent(backbone,p,case,language)
    if parent['action']!='edit':return parent_response,parent
    cached=extract(backbone,p,case,language)
    texture=torch.from_numpy(texture_features(ROOT/case['image']))
    return predict_cached(head,item,cached,texture,parent,policy)

"""Online R20: exactly the frozen R15 input encoder plus a registered CPU head."""
from .common import ROOT, EXP, PARENT, PARENT_SHA, ROUTES, protocol, sha
from .features import extract
from .controller import load_controller, predict_cached
from functional_v15.runtime import processor, verify_origin
from functional_v15.outcomes import *

def load_model(route, adapter=None):
    from functional_v15.runtime import load_model as load_backbone
    assert route in ROUTES and adapter is not None
    net,item = load_controller(adapter)
    assert item['route'] == route and item['parent_sha256'] == PARENT_SHA
    assert item['protocol_sha256'] == sha(EXP/'training-protocol.json')
    protocol()
    backbone = load_backbone('language_merger', PARENT)
    return backbone,net,item

def predict(model,p,case,language):
    backbone,net,item=model
    return predict_cached(net,item,extract(backbone,p,case,language))

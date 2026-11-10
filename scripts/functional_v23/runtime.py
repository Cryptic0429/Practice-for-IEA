"""Explicit parent binding inside this disposable process; no old source mutation."""
from .common import PARENT,PARENT_SHA
from functional_v13 import runtime as loader
from functional_v15.outcomes import input_row,parse
from functional_v1.runtime import processor,encode,generate
from functional_v11.runtime import build_batch

def load_model(route='language_merger',adapter=None,train=False,seed=None):
 assert route=='language_merger'
 loader.PARENT=PARENT;loader.PARENT_SHA=PARENT_SHA
 return loader.load_model(route,adapter=adapter,train=train,seed=seed)

def predict(model,p,case,language):
 response=generate(model,p,encode(p,input_row(case,language)),max_tokens=256)
 return response,parse(response)

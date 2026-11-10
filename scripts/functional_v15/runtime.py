"""R15 uses the exact R13 narrow model path and unchanged R11 literal prompt.

Only reviewed JSON labels and the independent shadow editor change. Reusing
the loader avoids silently introducing a precision, vision or adapter change.
"""
from functional_v13 import runtime as previous
from functional_v1.common import ROOT
from .outcomes import input_row, parse

EXP=ROOT/'experiments/functional-v15-20261106'
DATA=ROOT/'data/training/functional-v15-20261106'
PARENT=previous.PARENT
PARENT_SHA=previous.PARENT_SHA
ROUTES=('language_merger',)
processor=previous.processor
build_batch=previous.build_batch
verify_origin=previous.verify_origin
create_adapter=previous.create_adapter
transfer_parent=previous.transfer_parent
set_trainable=previous.set_trainable
expected_count=previous.expected_count


def load_model(route, adapter=None, train=False, seed=None):
    assert route in ROUTES
    return previous.load_model(route, adapter=adapter, train=train, seed=seed)


def predict(model, p, case, language):
    response=previous.generate(model,p,previous.encode(p,input_row(case,language)),max_tokens=256)
    return response,parse(response)

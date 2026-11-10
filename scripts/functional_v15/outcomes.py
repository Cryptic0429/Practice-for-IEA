"""Independent shadow-tool candidate; same literal prompt and strict JSON contract."""
import numpy as np
from functional_v11.outcomes import (validate_decision,parse,target_text,system_prompt,input_row,SCHEMA_VERSION,DECLINE_REASONS)

EDITOR_VERSION='functional-editor-v15-shared-shadow-gain-candidate'


def execute(rgb,decision):
    decision=validate_decision(decision);identity=dict(schema=SCHEMA_VERSION,editor=EDITOR_VERSION)
    if decision['action']=='decline':
        return dict(identity,status='declined',reason_code=decision['reason'],parameters=None,result=None)
    if decision['action']=='keep':
        return dict(identity,status='unchanged',reason_code=None,parameters={},result=rgb.copy())
    from .editor import edit
    result=edit(rgb,decision['parameters'],'bounded-fixed')
    if np.array_equal(rgb,result):raise ValueError('edit_produced_no_pixel_change')
    return dict(identity,status='edited',reason_code=None,parameters=decision['parameters'],result=result)

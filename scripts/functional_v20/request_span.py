"""Locate literal user-request tokens in an unchanged one-image Qwen prefix."""
def rendered_request(p,row,instruction):
    from functional_v1.runtime import messages
    assert instruction and row['conversations'][-1]['role']=='user'
    text=p.apply_chat_template(messages(row),tokenize=False,add_generation_prompt=True)
    start=text.rfind(instruction);assert start>=0
    end=start+len(instruction)
    tokenized=p.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    ids=tokenized['input_ids'];offsets=tokenized['offset_mapping']
    image_id=p.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    image_positions=[i for i,t in enumerate(ids) if t==image_id]
    assert len(image_positions)==1,'Exactly one source image is supported'
    selected=[i for i,(a,b) in enumerate(offsets) if b>start and a<end and b>a]
    assert selected and selected==list(range(selected[0],selected[-1]+1))
    assert selected[0]>image_positions[0]
    # The last selected token must precede the assistant generation suffix.
    assert offsets[selected[-1]][1]<=end
    return ids,selected,image_positions[0],image_id

def request_indices(p,row,instruction,encoded):
    base,selected,image_position,image_id=rendered_request(p,row,instruction)
    actual=encoded['input_ids'][0].tolist()
    count=actual.count(image_id);assert count>=1
    expected=base[:image_position]+[image_id]*count+base[image_position+1:]
    assert actual==expected,'Actual model prefix differs from the target-free rendered template'
    result=[i+count-1 for i in selected]
    assert [actual[i] for i in result]==[base[i] for i in selected]
    assert result[-1]<len(actual)-1
    return result

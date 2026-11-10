"""CPU actual tiny Qwen2.5-VL + PEFT test: neutral hook, gradients and one-file reload."""
import copy,json,tempfile
from pathlib import Path
import torch
from peft import LoraConfig,get_peft_model,PeftModel,get_peft_model_state_dict
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from .bridge import install_before_peft,set_features

def main():
    torch.set_num_threads(2);torch.manual_seed(20261131)
    config=Qwen2_5_VLConfig(text_config=dict(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,rope_scaling={'rope_type':'default','mrope_section':[2,1,1]}),
        vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,temporal_patch_size=2,
            spatial_merge_size=2,window_size=8,out_hidden_size=32,fullatt_block_indexes=[0]),
        image_token_id=3,video_token_id=6,vision_start_token_id=4,vision_end_token_id=5)
    base=Qwen2_5_VLForConditionalGeneration(config).eval();original=copy.deepcopy(base.state_dict())
    inputs=dict(input_ids=torch.tensor([[4,3,3,3,3,5,7,8]]),attention_mask=torch.ones(1,8,dtype=torch.long),
                pixel_values=torch.randn(16,24),image_grid_thw=torch.tensor([[1,4,4]]),labels=torch.tensor([[-100]*6+[7,8]]))
    with torch.no_grad():plain=base(**inputs).logits.clone()
    install_before_peft(base,32)
    model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,lora_dropout=0,bias='none',task_type='CAUSAL_LM',
         target_modules=r'.*(?:self_attn\.(?:q_proj|v_proj)|visual\.merger\.mlp\.(?:0|2))$',modules_to_save=['stat_bridge']))
    model.eval();values=torch.linspace(-.3,.4,16);set_features(model,values)
    with torch.no_grad():neutral=model(**inputs).logits
    assert torch.equal(neutral,plain), 'Zero-output bridge changed initial function'
    state=get_peft_model_state_dict(model)
    assert len(state)==16 and sum('stat_bridge' in n for n in state)==4
    active={n:w for n,w in model.named_parameters() if w.requires_grad}
    assert len(active)==16 and sum('modules_to_save.default' in n for n in active)==4
    model.train();optimizer=torch.optim.AdamW(list(active.values()),lr=.01,weight_decay=0)
    output=model(**inputs);output.loss.backward()
    assert all(w.grad is not None and torch.isfinite(w.grad).all() for w in active.values())
    assert all(w.grad.count_nonzero()==0 for n,w in active.items() if 'stat_bridge' in n and '.first.' in n)
    assert any(w.grad.count_nonzero()>0 for n,w in active.items() if 'stat_bridge' in n and '.last.' in n)
    optimizer.step();optimizer.zero_grad();model.eval()
    with torch.no_grad():trained=model(**inputs).logits.clone()
    set_features(model,-values)
    with torch.no_grad():changed=model(**inputs).logits
    assert not torch.equal(trained,changed), 'Current-image feature changes had no model path'
    set_features(model,values)
    with tempfile.TemporaryDirectory() as temp:
        model.save_pretrained(temp)
        saved=json.loads((Path(temp)/'adapter_config.json').read_text());assert saved['modules_to_save']==['stat_bridge']
        second=Qwen2_5_VLForConditionalGeneration(config);second.load_state_dict(original);install_before_peft(second,32)
        reloaded=PeftModel.from_pretrained(second,temp,is_trainable=False).eval();set_features(reloaded,values)
        assert len(reloaded.peft_config)==1 and len(get_peft_model_state_dict(reloaded))==16
        with torch.no_grad():restored=reloaded(**inputs).logits
        assert torch.equal(trained,restored), 'Saved bridge or adapter did not reload exactly'
    result=dict(status='CPU_tiny_actual_Qwen_PEFT_checks_passed',zero_output_logits_exact=True,one_adapter=True,
        bridge_saved_in_adapter_safetensors=True,changed_features_change_logits=True,first_layer_zero_initial_gradient_expected=True,
        real_7B_GPU_loaded=False,training_rows=0,quality_or_generalization_claim=False)
    root=Path(__file__).resolve().parents[2];out=root/'experiments/functional-v19-20261107/bridge-contract-check.json'
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()

"""Reload a saved training pilot and compare tensors and a deterministic probe."""
import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import set_seed
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer, SFTDataCollatorWith4DAttentionMask
from benchmark_stage1 import eval_probe


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    cli=parser.parse_args()
    source=json.loads((cli.directory/'benchmark.json').read_text())
    config=yaml.safe_load((cli.directory/'run-config.yaml').read_text())
    config['adapter_name_or_path']=str(cli.directory)
    config['max_samples']=1
    config['output_dir']=str(cli.directory.parent/(cli.directory.name+'-reload-check'))
    ma,da,ta,fa,ga=get_train_args(config)
    set_seed(ta.seed)
    tm=load_tokenizer(ma)
    template=get_template_and_fix_tokenizer(tm['tokenizer'],da)
    ds=get_dataset(template,ma,da,ta,stage='sft',**tm)['train_dataset']
    # Match training precision/upcast choices; no optimizer is constructed or run.
    model=load_model(tm['tokenizer'],ma,fa,is_trainable=True)
    state=get_peft_model_state_dict(model)
    saved=load_file(str(cli.directory/'adapter_model.safetensors'))
    assert set(state)==set(saved)
    for name,tensor in state.items(): assert torch.equal(tensor.detach().cpu(),saved[name]),name
    collator=SFTDataCollatorWith4DAttentionMask(template=template,model=model,pad_to_multiple_of=8,
                label_pad_token_id=-100,attn_implementation='sdpa',compute_dtype=ma.compute_dtype,**tm)
    actual=eval_probe(model,collator,ds)
    expected=source['saved_adapter_probe']
    assert actual['input_ids_sha256']==expected['input_ids_sha256']
    logits_delta=float((torch.tensor(actual['logits'])-torch.tensor(expected['logits'])).abs().max())
    loss_delta=abs(actual['loss']-expected['loss'])
    assert torch.allclose(torch.tensor(actual['logits']),torch.tensor(expected['logits']),atol=1e-4,rtol=1e-4)
    assert loss_delta<1e-4
    report=dict(adapter_reload_verified=True,adapter_tensors_exactly_equal=len(saved),
                absolute_loss_difference=loss_delta,max_logit_difference=logits_delta,
                training_updates_during_reload=0)
    (cli.directory/'reload-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('ADAPTER_RELOAD_PASSED',json.dumps(report),flush=True)


if __name__=='__main__':main()

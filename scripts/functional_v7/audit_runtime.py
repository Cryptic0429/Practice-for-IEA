"""CPU/meta-only proof of exact parent transfer into both adapter layouts."""
import gc
import os
from pathlib import Path

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration
from functional_v1.common import ROOT, BASE, write, sha, stamp
from .diagnose_editor import EXP
from .runtime import ROUTES, create_adapter, transfer_parent, set_trainable


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'This audit must not allocate a GPU'
    target = EXP / 'runtime-transfer-audit.json'
    assert not target.exists()
    parent = ROOT / 'models/from-functional-v2/sft-native-coverage/seed-01'
    assert sha(parent / 'adapter_model.safetensors') == '39d0e3b91c321ee0abeca8f3260116c76405d371127ba118e04822701e09f66c'
    config = AutoConfig.from_pretrained(BASE)
    results = []
    for route in ROUTES:
        with init_empty_weights():
            base = Qwen2_5_VLForConditionalGeneration(config)
            model = create_adapter(base, route)
        # Only the tiny LoRA modules are materialised. All base tensors stay meta.
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and ('.lora_A.' in name or '.lora_B.' in name):
                module.to_empty(device='cpu')
                if '.lora_B.' in name:
                    torch.nn.init.zeros_(module.weight)
                else:
                    torch.nn.init.kaiming_uniform_(module.weight, a=5 ** .5)
        receipt = transfer_parent(model, parent, route)
        names = set_trainable(model, route)
        assert all(value.is_meta for name, value in model.named_parameters() if 'lora_' not in name)
        assert all(not value.is_meta and value.dtype == torch.float32 for name, value in model.named_parameters() if name in names)
        receipt.update(trainable_tensors=len(names), all_base_parameters_meta=True,
                       active_parameter_count=sum(v.numel() for v in model.parameters() if v.requires_grad),
                       forward_or_gradient_test_performed=False)
        results.append(receipt)
        del model, base
        gc.collect()
    write(target, dict(at=stamp(), status='CPU_meta_parent_transfer_verified_only',
                       parent_sha256=sha(parent / 'adapter_model.safetensors'),
                       runtime_sha256=sha(Path(__file__).with_name('runtime.py')), results=results,
                       GPU_hours=0, new_training_started=False,
                       limitation='No quantized GPU load, forward, finite-gradient or trained-model capability proof.'))
    print('Exact 112-tensor parent transfer verified for both layouts; 4 zero-B merger tensors added in the expanded route. CPU/meta only.')


if __name__ == '__main__':
    main()

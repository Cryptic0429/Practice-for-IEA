"""Allow only the locally generated, hash-bound BatchFeature cache container."""
import argparse
import torch
from transformers.feature_extraction_utils import BatchFeature
from .common import *

def main():
 p=protocol();assert sha(ROOT/p['cache_path'])==p['cache_sha256']
 with torch.serialization.safe_globals([BatchFeature]):
  if '--preflight' in __import__('sys').argv:
   batches=torch.load(ROOT/p['cache_path'],map_location='cpu',weights_only=True)
   assert len(batches)==1958
   assert all(all(isinstance(v,torch.Tensor) and v.device.type=='cpu' for v in b.values()) for b in batches)
   assert not torch.cuda.is_initialized()
   write(EXP/'cache-container-preflight.json',dict(at=stamp(),status='passed_before_any_GPU_training',protocol_sha256=sha(EXP/'training-protocol.json'),cache_sha256=p['cache_sha256'],entry_sha256=sha(__file__),rows=1958,
    reason='108 newly built CPU entries retain the processor BatchFeature wrapper. First launch failed during weights_only load before GPUJob/model creation. Explicitly allowlist that owned wrapper; no weights_only=False, no source/cache/protocol modification, no learning/input change. Original failed logs retained.'))
   print('1958 hash-bound CPU batches loaded with exactly BatchFeature allowlisted; no GPU model loaded')
  else:
   from .train import main as train
   train()
if __name__=='__main__':main()

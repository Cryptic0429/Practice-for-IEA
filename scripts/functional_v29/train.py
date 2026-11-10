"""Fit CPU-only routing probes and LoRA on the final frozen Qwen decoder block."""
from collections import Counter
import time

import numpy as np
import torch
from torch import nn
from torch.nn.functional import linear,cross_entropy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2RMSNorm,Qwen2_5_VLDecoderLayer,Qwen2_5_VLRotaryEmbedding,
)

from functional_v1.common import ROOT,read,sha,stamp,write
from functional_v27.qwen_text import BASE
from .cache import EXP,PLAN,TRAIN,rows_and_plan

CACHE = EXP/'cache'
RUN = EXP/'run'


def load_cache():
    path = CACHE/'manifest.json'
    manifest = read(path)
    if (manifest['protocol_sha256']!=sha(PLAN)
        or manifest['base_config_sha256']!=sha(BASE/'config.json')
        or manifest['base_index_sha256']!=sha(BASE/'model.safetensors.index.json')):
        raise ValueError('Frozen cache identity mismatch')
    entries = []
    for b in manifest['blocks']:
        p = ROOT/b['path']
        if sha(p)!=b['sha256'] or b['start']!=len(entries):
            raise ValueError('Bad cache block')
        data = torch.load(p,map_location='cpu',weights_only=True)
        for i,length in enumerate(data['lengths'].tolist()):
            entries.append((data['prefix'][i,:length],data['final'][i],data['scores'][i]))
    if len(entries)!=manifest['rows']:
        raise ValueError('Wrong number of cached activations')
    return entries,manifest


class LoRALinear(nn.Module):
    def __init__(self,base,rank,alpha):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank,base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features,rank))
        self.scale = alpha/rank
        nn.init.kaiming_uniform_(self.a,a=5**.5)

    def forward(self,x):
        return self.base(x)+self.scale*linear(linear(x,self.a),self.b)


class Tail(nn.Module):
    def __init__(self,manifest):
        super().__init__()
        path = ROOT/manifest['tail_path']
        if sha(path)!=manifest['tail_sha256']:
            raise ValueError('Frozen final layer changed')
        weights = torch.load(path,map_location='cpu',weights_only=True)
        cfg = AutoConfig.from_pretrained(BASE,local_files_only=True)
        cfg._attn_implementation = 'sdpa'
        self.layer = Qwen2_5_VLDecoderLayer(cfg,weights['tail_index'])
        self.norm = Qwen2RMSNorm(cfg.hidden_size,eps=cfg.rms_norm_eps)
        self.rope = Qwen2_5_VLRotaryEmbedding(config=cfg)
        self.layer.load_state_dict(weights['layer'])
        self.norm.load_state_dict(weights['norm'])
        self.register_buffer('choices',weights['choices'])
        self.requires_grad_(False)
        self.eval()

    def add_lora(self,rank,alpha):
        for key in ('q_proj','v_proj'):
            projection = getattr(self.layer.self_attn,key)
            setattr(self.layer.self_attn,key,LoRALinear(projection,rank,alpha))

    def forward(self,sequences):
        lengths = torch.tensor([len(s) for s in sequences])
        states = nn.utils.rnn.pad_sequence(sequences,batch_first=True)
        pos = torch.arange(states.shape[1]).view(1,1,-1).expand(3,len(sequences),-1)
        cos_sin = self.rope(states,pos)
        # All padding is on the right. The causal mask alone protects valid positions.
        y = self.layer(states,position_ids=pos,use_cache=False,position_embeddings=cos_sin)[0]
        final = self.norm(y[torch.arange(len(sequences)),lengths-1])
        return linear(final,self.choices)


def fit_head(entries,indices,labels):
    x = torch.stack([entries[i][1] for i in indices]).numpy()
    y = np.array([labels[i] for i in indices])
    scale = StandardScaler().fit(x)
    classifier = LogisticRegression(C=.3,solver='lbfgs',max_iter=1000,random_state=20261107)
    counts = Counter(y)
    weights = np.array([1/np.sqrt(counts[i]) for i in y])
    weights /= weights.mean()
    classifier.fit(scale.transform(x),y,sample_weight=weights)
    if classifier.classes_.tolist()!=list(range(5)):
        raise ValueError('Missing route supervision in a fit condition')
    return {name:torch.as_tensor(v) for name,v in
            [('mean',scale.mean_),('scale',scale.scale_),('weight',classifier.coef_),('bias',classifier.intercept_)]}


def main():
    rows,plan = rows_and_plan()
    torch.set_num_threads(plan['cpu_threads'])
    torch.set_num_interop_threads(1)
    entries,manifest = load_cache()
    if RUN.exists():
        raise FileExistsError('Keep first fitted run immutable')
    labels_by_name = {name:i for i,name in enumerate(plan['labels'].values())}
    labels = [labels_by_name[r['expected']] for r in rows]
    legacy = [i for i,r in enumerate(rows) if r['cohort']=='legacy']
    train = [i for i,r in enumerate(rows) if r['cohort'] in TRAIN]
    model = Tail(manifest)
    cfg = plan['lo_ra']
    with torch.inference_mode():
        parity = max(float((model([e[0] for e in entries[s:s+cfg['batch_size']]]) -
                            torch.stack([e[2] for e in entries[s:s+cfg['batch_size']]])).abs().max())
                     for s in range(0,len(entries),cfg['batch_size']))
    if parity>1e-3:
        raise ValueError(f'Cached tail does not replay base logits: {parity}')
    print('Exact-tail replay max logit error:',parity,flush=True)
    RUN.mkdir()
    torch.save(dict(legacy=fit_head(entries,legacy,labels),augmented=fit_head(entries,train,labels)),RUN/'heads.pt')
    torch.manual_seed(cfg['seed'])
    model.add_lora(cfg['rank'],cfg['alpha'])
    adapter = {name:p for name,p in model.named_parameters() if p.requires_grad}
    if len(adapter)!=4:
        raise ValueError('Unexpected trainable set')
    optimizer = torch.optim.AdamW(adapter.values(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    counts = torch.bincount(torch.tensor([labels[i] for i in train]),minlength=5)
    weight = counts.float().rsqrt()
    metrics = []
    start = time.monotonic()
    generator = torch.Generator().manual_seed(cfg['seed'])
    steps = 0
    for epoch in range(1,cfg['epochs']+1):
        order = torch.randperm(len(train),generator=generator).tolist()
        batches = []
        for j in range(0,len(order),cfg['batch_size']):
            ids = [train[n] for n in order[j:j+cfg['batch_size']]]
            logits = model([entries[i][0] for i in ids])
            loss = cross_entropy(logits,torch.tensor([labels[i] for i in ids]),weight=weight)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.values(),1.)
            optimizer.step()
            batches.append(float(loss.detach()))
            steps += 1
        metrics.append(dict(epoch=epoch,step=steps,mean_train_loss=float(np.mean(batches))))
        write(RUN/'progress.json',dict(at=stamp(),metrics=metrics))
        print(metrics[-1],flush=True)
    params = {key:v.detach().clone() for key,v in adapter.items()}
    if any(not torch.count_nonzero(v) for key,v in params.items() if key.endswith('.b')):
        raise ValueError('LoRA did not train both projections')
    torch.save(dict(parameters=params,rank=cfg['rank'],alpha=cfg['alpha'],
                    tail_sha256=manifest['tail_sha256'],protocol_sha256=sha(PLAN)),RUN/'adapter.pt')
    write(RUN/'receipt.json',dict(at=stamp(),protocol_sha256=sha(PLAN),cache_sha256=sha(CACHE/'manifest.json'),
        adapter_sha256=sha(RUN/'adapter.pt'),heads_sha256=sha(RUN/'heads.pt'),cpu_only=not torch.cuda.is_initialized(),
        legacy_rows=len(legacy),fit_rows=len(train),trainable_parameters=sum(v.numel() for v in params.values()),
        tail_replay_max_logit_error=parity,steps=steps,seconds=time.monotonic()-start,metrics=metrics,GPU_hours=0))


if __name__ == '__main__':
    main()

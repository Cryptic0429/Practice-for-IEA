"""CPU residual features; exact R20 frozen hidden cache is an explicit dependency."""
from .common import ROOT,EXP,protocol,read,write,sha,stamp

def main():
    p=protocol()
    import torch
    from .texture import features,NAMES
    torch.set_num_threads(2)
    paths={}
    for split in ('train','old_native','native_development','decision_development'):
        source=p['sources'][split];assert sha(ROOT/source['path'])==source['sha256']
        for c in read(ROOT/source['path']):
            assert sha(ROOT/c['image'])==c['input_sha256']
            if c['image'] not in paths:
                paths[c['image']]=dict(input_sha256=c['input_sha256'],values=features(ROOT/c['image']).tolist())
    write(EXP/'texture-features.json',dict(at=stamp(),names=list(NAMES),images=paths,
        implementation_sha256=sha(ROOT/'scripts/functional_v21/texture.py'),target_files_read=False,
        notes='Median-filter residuals contain real texture/edges as well as noise. These are learned-head inputs, never automatic visual quality labels.'))
    assert not torch.cuda.is_initialized()
    print('CPU fixed current-input observations',len(paths),'paths; noGPU or target reads.')

if __name__=='__main__':main()

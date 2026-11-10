"""Frozen external denoiser on four training-only capacity/detail controls."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from functional_v1.common import ROOT,read,write,sha,stamp,GPUJob
from functional_v1.editor import load_rgb,save
from functional_v1.curate import font
from .diagnose_editor import EXP

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',choices=['0','2','3'],required=True);a=ap.parse_args()
    out=EXP/'learned-denoise-feasibility';assert not (out/'protocol.json').exists()
    source_protocol=EXP/'denoise-feasibility/protocol.json';sources=read(source_protocol)['sources']
    directory=ROOT/'models/external/SCUNet';metadata=read(directory/'metadata.json')
    weight=directory/'scunet_color_real_psnr.pth';wheel=directory/'einops-0.8.1-py3-none-any.whl'
    assert sha(weight)==metadata['weight_sha256'] and sha(wheel)==metadata['einops_wheel_sha256']
    assert sha(ROOT/'scripts/functional_v7/scunet_network.py')==metadata['compatibility_code_sha256']
    assert sha(ROOT/'third_party/SCUNet/models/network_scunet.py')==metadata['upstream_code_sha256']
    for r in sources:assert sha(ROOT/r['input'])==r['input_sha256'] and sha(ROOT/r['target'])==r['target_sha256']
    write(out/'protocol.json',dict(at=stamp(),status='frozen_before_any_learned_denoising_output',sources=sources,
        source_protocol_sha256=sha(source_protocol),external_model=metadata,code_sha256=sha(__file__),
        inference='Official SCUNet config[4]*7 dim64 in_nc3, strict full state load, FP32 RGB0-1, original resolution, official internal replication-pad to64. No tiling or image resizing.',
        order='After the EXACT old adopted/rejected tonal target; no tone changes in this probe.',strengths=[0,50,100],
        blending='Output is original target + strength/100*(clamped learned denoise - original target); zero is exact original pixels.',
        data='Only four original training-pool images, same noisy source and three texture controls as classical NLM. No development or acceptance inference.',
        learning='External published pretrained denoiser, no project training/fine-tuning. Effect cannot count as instruction-model learning.',
        decision='Direct anonymous full-frame/detail review only; no automatic target admission or live-tool registration.',budget='Same original16 occupied GPUh, assigned GPU only, ten-minute job cap; GPU1 reserved.'))
    sys.path.insert(0,str(wheel))
    import torch
    from .scunet_network import SCUNet
    torch.set_num_threads(4);torch.manual_seed(20261112)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    rows=[];sheets=[]
    try:
        with GPUJob(a.gpu,'v7-external-scunet-training-pool-capacity-probe',max_seconds=600) as budget:
            model=SCUNet(in_nc=3,config=[4]*7,dim=64,drop_path_rate=0)
            state=torch.load(weight,map_location='cpu',weights_only=True)
            assert all(torch.isfinite(v).all() for v in state.values())
            model.load_state_dict(state,strict=True);model.eval().requires_grad_(False);model.to('cuda')
            write(out/'load-verification.json',dict(at=stamp(),strict_state_dict=True,tensor_count=len(state),
                parameter_count=sum(p.numel() for p in model.parameters()),all_frozen=not any(p.requires_grad for p in model.parameters()),
                weights_only_safe_loading=True,external_weight_sha256=sha(weight),compatibility_only_import_changes=True))
            for r in sources:
                budget.tick();target=load_rgb(ROOT/r['target']);start=time.time()
                tensor=torch.from_numpy(target.copy()).permute(2,0,1).unsqueeze(0).float().div(255).to('cuda')
                with torch.inference_mode():clean=model(tensor)
                torch.cuda.synchronize();seconds=time.time()-start
                assert clean.shape==tensor.shape and torch.isfinite(clean).all()
                clean=clean[0].permute(1,2,0).clamp(0,1).cpu().numpy();entries=[]
                for strength in (0,50,100):
                    image=target.copy() if strength==0 else np.clip(np.rint((target.astype(np.float32)/255*(1-strength/100)+clean*(strength/100))*255),0,255).astype(np.uint8)
                    key=r['case_id']+'-'+hashlib.sha256(image.tobytes()).hexdigest()[:12];path=out/'images'/f'{key}.png';save(path,image)
                    row=dict(case_id=r['case_id'],item_id=key,strength=strength,image=str(path.relative_to(ROOT)),image_sha256=sha(path),
                        external_model_weight_sha256=metadata['weight_sha256'],inference_seconds=seconds,shape=list(target.shape),new_training_label_admitted=False)
                    rows.append(row);entries.append(row)
                entries.sort(key=lambda x:x['item_id']);canvas=Image.new('RGB',(1800,1320),'white');d=ImageDraw.Draw(canvas)
                d.text((12,10),r['case_id'],fill='black',font=font(24));d.text((12,50),r['instructions']['en'][:125],fill='black',font=font(22))
                for i,(label,path) in enumerate([('INPUT',ROOT/r['input'])]+[(x['item_id'][-12:],ROOT/x['image']) for x in entries]):
                    x,y=(i%3)*600,120+(i//3)*600;d.text((x+12,y),label,fill='black',font=font(22))
                    with Image.open(path) as im:fitted=ImageOps.contain(im.convert('RGB'),(580,550),Image.Resampling.LANCZOS)
                    canvas.paste(fitted,(x+10+(580-fitted.width)//2,y+40+(550-fitted.height)//2))
                path=out/(r['case_id']+'.jpg');canvas.save(path,quality=95,subsampling=0);sheets.append(dict(case_id=r['case_id'],sheet=str(path.relative_to(ROOT)),sha256=sha(path),items=[x['item_id'] for x in entries]))
                write(out/'status.json',dict(status='running',sources_completed=len(sheets),sources_total=4,at=stamp()));budget.tick()
        write(out/'manifest.json',dict(at=stamp(),records=rows,sheets=sheets,protocol_sha256=sha(out/'protocol.json')))
        write(out/'status.json',dict(status='completed_predictions_pending_direct_visual_review',sources_completed=4,sources_total=4,at=stamp(),training_started=False))
        print('Twelve training-only denoise variants generated; no target admission, live change or new training.')
    except BaseException as e:
        write(out/'status.json',dict(status='failed',error=f'{type(e).__name__}: {e}',at=stamp(),sources_completed=len(sheets)))
        raise

if __name__=='__main__':main()

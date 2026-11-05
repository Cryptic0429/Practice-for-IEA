"""Check image-patch/row alignment across generation concurrency settings, CPU only."""
from types import SimpleNamespace
import torch
from stage2_runtime import chunk_generation


class FakeModel:
    def __init__(self): self.calls=[]
    def generate(self,**kwargs):
        ids=kwargs['input_ids'];grid=kwargs['image_grid_thw'];pixels=kwargs['pixel_values']
        self.calls.append(len(ids));offset=0
        for row,g in zip(ids,grid):
            count=int(g.prod());assert torch.equal(pixels[offset:offset+count],row[0].expand(count,1))
            offset+=count
        assert offset==len(pixels)
        return torch.cat([ids,ids[:,:1]],dim=1)


def main():
    ids=torch.arange(16).view(16,1).expand(16,3)
    grid=torch.tensor([[1,1,1+i//4] for i in range(16)])
    pixels=torch.cat([torch.full((int(g.prod()),1),i) for i,g in enumerate(grid)])
    for concurrency in [1,2,4]:
        model=FakeModel();chunk_generation(model,concurrency)
        result=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),pixel_values=pixels,
                              image_grid_thw=grid,generation_config=SimpleNamespace(pad_token_id=0))
        assert torch.equal(result,torch.cat([ids,ids[:,:1]],dim=1))
        assert model.calls==[concurrency]*(16//concurrency)
        # Regression for left padding: the backend must receive only real tokens,
        # while the returned prefix retains the shape expected by TRL.
        padded=ids.clone();mask=torch.ones_like(ids)
        for group in range(4):
            n=group%3
            padded[group*4:(group+1)*4,:n]=-1
            mask[group*4:(group+1)*4,:n]=0
        model=FakeModel();chunk_generation(model,concurrency)
        result=model.generate(input_ids=padded,attention_mask=mask,pixel_values=pixels,
                              image_grid_thw=grid,generation_config=SimpleNamespace(pad_token_id=0))
        assert torch.equal(result,torch.cat([padded,ids[:,:1]],dim=1))
    print('PATCH_ALIGNMENT_AND_CANDIDATE_ORDER_OK:1,2,4')


if __name__=='__main__':
    main()

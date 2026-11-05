"""Assistant-only sequence DPO objective. References are frozen scalar log probabilities."""
import torch
import torch.nn.functional as F


def sequence_logp(logits,labels):
    shifted=labels[:,1:]
    mask=shifted.ne(-100)
    scores=logits[:,:-1,:][mask].float()
    targets=shifted[mask]
    if not targets.numel():raise ValueError('No supervised assistant tokens')
    return -F.cross_entropy(scores,targets,reduction='sum'),targets.numel()


def dpo_loss(chosen,rejected,ref_chosen,ref_rejected,beta=.1):
    margin=beta*((chosen-rejected)-(ref_chosen-ref_rejected))
    return -F.logsigmoid(margin)

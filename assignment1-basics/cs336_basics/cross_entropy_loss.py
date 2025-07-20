from jaxtyping import Float, Int
import torch as t
import torch.distributed as dist

def cross_entropy(logits: Float[t.Tensor, "b d"], target: Int[t.Tensor, "b"]) -> t.Tensor:
    # print("ce target, logits", dist.get_rank(), target.device, logits.device)
    log_probs = log_softmax(logits, dim=-1)
    # print("ce log_probs", dist.get_rank(), log_probs.device)
    return -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1).mean()

def log_softmax(logits: Float[t.Tensor, "b d"], dim: int):
    m = logits.max(dim=dim, keepdim=True).values
    v = logits - m
    return v - logsumexp(v, dim)

def logsumexp(x: t.Tensor, dim: int) -> t.Tensor:
    m = x.max(dim=dim, keepdim=True).values
    return m + t.log((x - m).exp().sum(dim=dim, keepdim=True))

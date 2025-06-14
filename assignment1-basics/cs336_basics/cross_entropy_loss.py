from jaxtyping import Float, Int
import torch as t

def cross_entropy(logits: Float[t.Tensor, "b d"], target: Int[t.Tensor, "b"]) -> t.Tensor:
    log_probs = log_softmax(logits, dim=-1)
    return -log_probs.gather(dim=-1, index=target.unsqueeze(1)).mean().squeeze()

def log_softmax(logits: Float[t.Tensor, "b d"], dim: int):
    m = logits.max(dim=dim, keepdim=True).values
    v = logits - m
    return v - t.log(v.exp().sum(dim=dim, keepdim=True))

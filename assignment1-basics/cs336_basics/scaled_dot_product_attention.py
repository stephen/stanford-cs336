import math
import einops
from jaxtyping import Float
import torch as t
from typing import Any, Optional

from cs336_basics.softmax import softmax
from torch.distributed.tensor import distribute_tensor, Replicate, Shard

def scaled_dot_product_attention(
    q: Float[t.Tensor, "... n d"],
    k: Float[t.Tensor, "... m d"],
    v: Float[t.Tensor, "... m d"],
    mask: Optional[Float[t.Tensor, "... n m"]],
) -> Float[t.Tensor, "n d"]:
    q_k = einops.einsum(q, k, "... n d, ... m d -> ... n m")
    d = k.shape[-1]

    # Note that mask = False means "ignore". It's more like an "allow mask".
    z = (q_k / math.sqrt(d))
    if mask is not None:
        add = t.where(mask == False, -math.inf, 0)
        z += add
    z = softmax(z, dim=-1)

    return einops.einsum(z, v, "... n m, ... m d -> ... n d")

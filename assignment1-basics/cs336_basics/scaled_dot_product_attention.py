import math
import einops
from jaxtyping import Float
import torch as t

from cs336_basics.softmax import softmax

def scaled_dot_product_attention(
    q: Float[t.Tensor, "b n d"],
    k: Float[t.Tensor, "b m d"],
    v: Float[t.Tensor, "b m d"],
    mask: Float[t.Tensor, "b n m"],
) -> Float[t.Tensor, "d d"]:
    q_k = einops.einsum(q, k, "... n d, ... m d -> ... n m")
    d = k.shape[-1]

    # Note that mask = False means "ignore". It's more like an "allow mask".
    pre_softmax_mask = t.where(mask == False, -math.inf, 0)

    z = softmax((q_k / math.sqrt(d)) + pre_softmax_mask, dim=-1)

    return einops.einsum(z, v, "... n m, ... m d -> ... n d")

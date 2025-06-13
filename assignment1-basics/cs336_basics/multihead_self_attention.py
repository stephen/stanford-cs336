import einops
from jaxtyping import Float
from typing import Optional, overload
import torch as t

from cs336_basics.linear import Linear
from cs336_basics.rope import RoPE
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadSelfAttention(t.nn.Module):
    @overload
    def __init__(self, d_model: int, n_heads: int, rope_theta: float, rope_max_seq_length: int,  device: Optional[t.device] = None): ...
    @overload
    def __init__(self, d_model: int, n_heads: int): ...

    def __init__(self, d_model: int, n_heads: int, rope_theta: Optional[float] = None, rope_max_seq_length: Optional[int] = None, device: Optional[t.device] = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        self.device = device

        assert ((rope_theta is None) == (rope_max_seq_length is None)), "rope_theta and rope_max_seq_length must both be specified or not"
        if rope_theta is not None and rope_max_seq_length is not None:
            self.rope = RoPE(rope_theta, self.d_h, rope_max_seq_length)
        else:
            self.rope = None

        self.Wq = Linear(d_model, self.d_h * n_heads, device=device)
        self.Wk = Linear(d_model, self.d_h * n_heads, device=device)
        self.Wv = Linear(d_model, self.d_h * n_heads, device=device)
        self.Wo = Linear(self.d_h * n_heads, d_model, device=device)

    def forward(self, x: Float[t.Tensor, "... n d"], token_positions: Optional[Float[t.Tensor, "n"]] = None) -> t.Tensor:
        assert ((token_positions is None) == (self.rope is None)), "token_positions can only be specified if rope parameters are specified"

        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        # Note that the input shape here should be (h d) not (d h) because the weights
        # given to use in the adapter are annotated as "d_k d_in", i.e. in our notation
        # "d_heads d_model".
        q = einops.rearrange(q, "... n (h d) -> ... h n d", h=self.n_heads)
        k = einops.rearrange(k, "... n (h d) -> ... h n d", h=self.n_heads)
        v = einops.rearrange(v, "... n (h d) -> ... h n d", h=self.n_heads)

        n = q.shape[-2]
        m = k.shape[-2]
        mask = t.tril(t.ones((n, m), device=self.device)).bool()

        if self.rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # This expects things to be in the shape [... n d].
        a = scaled_dot_product_attention(q, k, v, mask)

        a = einops.rearrange(a, "... h n d -> ... n (h d)")

        return self.Wo(a)

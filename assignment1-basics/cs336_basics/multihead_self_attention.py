import einops
from jaxtyping import Float, Bool
from typing import Optional
import torch as t

from cs336_basics.linear import Linear
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention

class MultiHeadSelfAttention(t.nn.Module):
    def __init__(self, d_model: int, n_heads: int, device: Optional[t.device] = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        self.device = device

        self.Wq = Linear(d_model, self.d_h * n_heads, device=device)
        self.Wk = Linear(d_model, self.d_h * n_heads, device=device)
        self.Wv = Linear(d_model, self.d_h * n_heads, device=device)
        self.Wo = Linear(self.d_h * n_heads, d_model, device=device)

    def forward(self, x: Float[t.Tensor, "... n d"]) -> t.Tensor:
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

        # This expects things to be in the shape [... n d].
        a = scaled_dot_product_attention(q, k, v, mask)

        a = einops.rearrange(a, "... h n d -> ... n (h d)")

        return self.Wo(a)

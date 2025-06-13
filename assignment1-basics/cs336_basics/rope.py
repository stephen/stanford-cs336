from typing import Optional, Tuple
import einops
import torch as t

class RoPE(t.nn.Module):
    cos_cached: t.Tensor
    sin_cached: t.Tensor

    def __init__(self, theta: float, d_k: int, max_seq_length: int, device: Optional[t.device] = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_length = max_seq_length
        self.device = device
        c, s = self._precompute()

        self.register_buffer("cos_cached", c, persistent=False)
        self.register_buffer("sin_cached", s, persistent=False)

    def _precompute(self) -> Tuple[t.Tensor, t.Tensor]:
        positions = t.arange(self.max_seq_length, device=self.device, dtype=t.float16)
        dimensions = t.arange(0, self.d_k, step=2, device=self.device)

        # Instead of doing 2d/k, we use step size 2 for dimensions so the dimension is only d//2.
        freqs = 1.0 / (self.theta ** (dimensions / self.d_k))

        thetas = einops.einsum(freqs, positions, "dimension, position -> position dimension")

        return (thetas.cos(), thetas.sin())


    def forward(self, x: t.Tensor, token_positions: t.Tensor) -> t.Tensor:
        c = self.cos_cached[token_positions]
        s = self.sin_cached[token_positions]

        # In the last dimension (d_model), grab all of the even/odd indices.
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rv_even = c * x_even - s * x_odd
        rv_odd = s * x_even + c * x_odd

        rv = t.empty_like(x)
        rv[..., 0::2] = rv_even
        rv[..., 1::2] = rv_odd

        return rv

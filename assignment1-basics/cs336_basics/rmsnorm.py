from typing import Optional
from einops import einsum
import torch as t

class RMSNorm(t.nn.Module):
  def __init__(self, d_model: int, eps: float = 1e-5, device: Optional[t.device] = None, dtype: Optional[t.dtype] = None):
    super().__init__()

    self.gain = t.nn.Parameter(t.ones(d_model, device=device, dtype=dtype))
    self.eps = eps

  def forward(self, x: t.Tensor):
    # x is b, n, d
    in_dtype = x.dtype
    x = x.to(t.float32)

    d_model = self.gain.shape[0]

    rms = t.sqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)

    rv = x / rms * self.gain

    return rv.to(in_dtype)

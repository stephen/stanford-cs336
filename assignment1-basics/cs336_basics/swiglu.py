from typing import Optional
from einops import einsum
import torch as t


def silu(x: t.Tensor) -> t.Tensor:
  return t.sigmoid(x) * x

class SwiGLU(t.nn.Module):
  def __init__(self, d_model: int, d_ff: int, device: Optional[t.device] = None, dtype: Optional[t.dtype] = None):
    super().__init__()

    self.w1 = t.nn.Linear(d_model, d_ff, device=device, dtype=dtype)
    self.w2 = t.nn.Linear(d_ff, d_model, device=device, dtype=dtype)
    self.w3 = t.nn.Linear(d_model, d_ff, device=device, dtype=dtype)

  def forward(self, x: t.Tensor):
    gated = silu(self.w1(x))
    up_projected = self.w3(x)
    return self.w2(gated * up_projected)

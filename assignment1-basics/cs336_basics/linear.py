import math
from typing import Optional
import einops
import torch as t

class Linear(t.nn.Module):
  def __init__(self, in_features: int, out_features: int, device: Optional[t.device] = None, dtype: Optional[t.dtype] = None):
    super().__init__()

    w = t.zeros((out_features, in_features), device=device, dtype=dtype)
    variance = 2 / (in_features + out_features)
    t.nn.init.trunc_normal_(w, 0, math.sqrt(variance), -3, 3)

    self.w = t.nn.Parameter(w)

  def forward(self, input: t.Tensor):
    return einops.einsum(input, self.w, "... input, output input -> ... output")

import math
from typing import Iterable
import torch as t

def clip_gradients(params: Iterable[t.nn.Parameter], max_l2_norm: float) -> None:
    l2_norm = math.sqrt(sum([t.sum(p.grad**2) for p in params if p.grad is not None]))

    if l2_norm < max_l2_norm:
        return

    eps = 1e-6
    scale = max_l2_norm / (l2_norm + eps)
    for p in params:
        if p.grad is not None:
            p.grad *= scale

from typing import Optional
import torch as t

class Embedding(t.nn.Module):
  def __init__(self, n_embed: int, d_embed: int, device: Optional[t.device] = None, dtype: Optional[t.dtype] = None):
    super().__init__()

    embedding = t.zeros((n_embed, d_embed), device=device, dtype=dtype)
    t.nn.init.trunc_normal_(embedding, 0, 1, -3, 3)

    self.embedding = t.nn.Parameter(embedding)

  def forward(self, token_ids: t.Tensor):
    return self.embedding[token_ids]

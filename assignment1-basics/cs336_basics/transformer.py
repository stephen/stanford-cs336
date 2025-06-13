import itertools
from typing import Optional
import torch as t

from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.tokenizer_cls import Tokenizer
from cs336_basics.transformer_block import Transformer
from functools import reduce

class TransformerLM(t.nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_len: int,
            n_layers: int,
            d_model: int,
            n_heads: int,
            d_ff: int,
            rope_theta: Optional[float] = None,
            device: Optional[t.device] = None
        ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device)
        self.layers = t.nn.ModuleList([Transformer(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            rope_max_seq_len=context_len if rope_theta else None,
            rope_theta=rope_theta,
            device=device,
        ) for _ in range(n_layers)])

        self.ln = RMSNorm(d_model, device=device)
        self.output = Linear(d_model, vocab_size, device=device)

    def forward(self, x: t.Tensor) -> t.Tensor:
        layers = itertools.chain([self.embedding], self.layers, [self.ln, self.output])
        return reduce(lambda x, layer: layer(x), layers, x)

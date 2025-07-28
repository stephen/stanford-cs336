import itertools
from typing import Any, Optional
import torch as t

from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.tokenizer_cls import Tokenizer
from cs336_basics.transformer_block import Transformer
from functools import reduce
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module, SequenceParallel, PrepareModuleInput
from torch.distributed.tensor import Replicate, Shard

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
            device: Optional[t.device] = None,
            mesh: Optional[Any] = None,
            tp = -1,
            loss_parallel: Optional[bool] = False,
        ):
        super().__init__()
        self.context_len = context_len
        self.embedding = t.nn.Embedding(vocab_size, d_model, device=device)
        self.mesh = mesh
        self.loss_parallel = loss_parallel

        self.layers = t.nn.ModuleList([Transformer(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            rope_max_seq_len=context_len if rope_theta else None,
            rope_theta=rope_theta,
            device=device,
            tp=tp,
            mesh=mesh,
        ) for _ in range(n_layers)])

        self.ln = RMSNorm(d_model, device=device)
        self.output = t.nn.Linear(d_model, vocab_size, device=device)

        if self.mesh and tp > 1:
            local_attn_proj = True
            layer_tp_plan = {
                "attn.Wq": ColwiseParallel(use_local_output=local_attn_proj),
                "attn.Wk": ColwiseParallel(use_local_output=local_attn_proj),
                "attn.Wv": ColwiseParallel(use_local_output=local_attn_proj),
                # attn_out = [n, d_c]
                "attn.Wo": RowwiseParallel(output_layouts=Shard(1)),
                "attn": PrepareModuleInput(
                    input_layouts=[Shard(1), Replicate()],
                    desired_input_layouts=(Replicate(), Replicate()),
                ),
                "ln1": SequenceParallel(),
                "ffn": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "ffn.w1": ColwiseParallel(),
                "ffn.w2": RowwiseParallel(output_layouts=Shard(1)),
                "ffn.w3": ColwiseParallel(),
                "ln2": SequenceParallel(),
            }
            for layer in self.layers:
                parallelize_module(layer, parallelize_plan=layer_tp_plan, device_mesh=self.mesh["tp"])

            # print("loss parallel?", self.loss_parallel)
            transformer_tp_plan = {
                "embedding": RowwiseParallel( # vocab parallel E_w = [V, D]
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                ),
                "ln": SequenceParallel(),
                "output": ColwiseParallel( # vocab parallel O_w = [D, V]
                    input_layouts=Shard(1),
                    output_layouts=(Shard(2) if self.loss_parallel else Replicate()),
                    use_local_output=(False if self.loss_parallel else True),
                ),
            }
            # print(transformer_tp_plan)
            parallelize_module(self, parallelize_plan=transformer_tp_plan, device_mesh=self.mesh["tp"])



    def forward(self, x: t.Tensor) -> t.Tensor:
        assert x.shape[-1] <= self.context_len, f"context_len cannot exceed max {self.context_len}, got {x.shape[-1]}"
        layers = itertools.chain([self.embedding], self.layers, [self.ln, self.output])
        return reduce(lambda x, layer: layer(x), layers, x)

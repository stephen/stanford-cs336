from typing import Any, Optional
import torch as t

from cs336_basics.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU

class Transformer(t.nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int,
            rope_theta: Optional[float] = None,
            rope_max_seq_len: Optional[int] = None,
            device: Optional[t.device] = None,
            tp: Optional[int] = None,
            mesh: Optional[Any] = None,
        ):
        super().__init__()
        self.device = device
        self.ln1 = RMSNorm(d_model, device=device)
        self.ffn = SwiGLU(d_model, d_ff, device=device)
        self.ln2 = RMSNorm(d_model, device=device)
        # self.attn = t.nn.MultiheadAttention(d_model, n_heads, 0, False, False, )
        self.attn = MultiHeadSelfAttention(
            d_model, n_heads, rope_theta=rope_theta,
            rope_max_seq_length=rope_max_seq_len,
            device=device,
            tp=tp,
            mesh=mesh,
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        token_positions = t.arange(x.shape[1], device=self.device)
        y = self.attn(self.ln1(x), token_positions) + x
        return self.ffn(self.ln2(y)) + y

    # sequence_parallel(x) [seq_len[rank], model_dim]
    # rms_norm(x) seq_parallel => [seq_len[rank], model_dim]
    # all_gather(x) [seq_len, model_dim]
    # FFN/attn (rowwise + colwise) [rank][seq_len, model]
    # reduce_scatter(y) [seq_len[rank], model]
    # y + res seq_parallel [seq_len[rank], model_dim]
    # rms_norm
    # all_gather

    # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
    # > Note: Executing ReduceScatter, followed by AllGather, is equivalent to the AllReduce operation.

# all_reduce = reduce-scatter + all-gather

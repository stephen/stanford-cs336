from typing import Optional
from einops import einsum
import torch as t

from cs336_basics.linear import Linear

def silu(x: t.Tensor) -> t.Tensor:
  return t.sigmoid(x) * x

# [m, k] @ [n, k]
# [i, :] Rowwise
# [:, j] Column wise
# X @ w13[n_chunk * 2, k] => w13_out[m, n_chunk * 2] Rowwise
# w13[m, ith_chunk] = concat(w1[m, ith_chunk], w3[m, ith_chunk])
# w13_out[m, n_chunk] @ w2[d_model, n_chunk] => [m, d_model] Columnwise
# sum_ranks [m, d_model]

# x @ x1 (by rank)
# xw1_0 xw1_1
# xw3_0 xw3_1

# X
# 0,0  0,1   0,2
#
#
#y = xw
# y[i, j] = sum_k (x[i, k] * w[j, k])
def matmul(x, w13, w2):
    # Physical layouts, assume [m, k] @ [n, k]
    # Logical layout is [m, k] @ [k, n]
    m, k = x.shape
    n, k = w13.shape
    k, n = w2.shape

    y = zeros(m, n)

    # all gather(x)
    for rank in range(tp_size):
        chunk_size = n // tp_size
        # Rowwise
        for j in range(rank * chunk_size, (rank + 1) * chunk_size):
            for i in range(m):
                for kk in range(k):
                    y[i, j] += x[i, kk] * w[j, kk]

        # y[:, rank*chunk_size:(rank+1)*chunk_size] is complete
        # y[m, chunk] @ w2[k, chunk]

        # Columnwise
        for i in range(m):
            for kk in range(k):
                for chunk_j in range(rank * chunk_size, (rank + 1) * chunk_size):
                    out[rank][i, kk] += y[i, chunk_j] * w2[kk, chunk_j]
    # all_reduce(out)
    for rank in ranks:
        out_global[:, :] += out[rank][:, :]


class SwiGLUParallel(t.nn.Module):
  def __init__(self, d_model: int, d_ff: int, device: Optional[t.device] = None, dtype: Optional[t.dtype] = None):
    super().__init__()

    # X[m, d_model] @ [d_model, d_ff]
    # w13_out = X[m, d_model] @ (concat(w1, w3, axis=-1))[d_model, d_ff * 2]
    # act = silu(w13_out[:, :d_ff]) * w13_out[:, d_ff:]
    # out = act[m, d_ff] @ w2[d_ff, d_model]
    self.w1 = Linear(d_model, d_ff, device, dtype)
    self.w2 = Linear(d_ff, d_model, device, dtype)
    self.w3 = Linear(d_model, d_ff, device, dtype)

  def forward(self, x: t.Tensor):
    gated = silu(self.w1(x))
    up_projected = self.w3(x)
    return self.w2(gated * up_projected)

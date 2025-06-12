import torch as t

def softmax(x: t.Tensor, dim: int) -> t.Tensor:
    m = t.max(x, dim=dim, keepdim=True).values
    v_exp = (x - m).exp()
    return v_exp / v_exp.sum(dim=dim, keepdim=True)

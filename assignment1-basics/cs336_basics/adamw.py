import torch as t
from typing import Callable, Optional
from torch import Tensor

class AdamW(t.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay: float = 0.01, betas: tuple[float, float] = (.9, .999), eps: float = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> float | None: # type: ignore
        loss = None if closure is None else closure()
        # param_groups is useful in case different params use different learning hyperparams.
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.

            weight_decay = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                if 'm' not in state:
                    state['m'] = t.zeros_like(p)
                m: Tensor = state.get("m")

                if 'v' not in state:
                    state['v'] = t.zeros_like(p)
                v: Tensor = state.get("v")

                time = state.get("t", 1)

                grad = p.grad.data

                m.copy_(beta1 * m + (1 - beta1) * grad)
                v.copy_(beta2 * v + (1 - beta2) * grad**2)

                lr_t = lr * (1 - beta2**time) ** 0.5 / (1 - beta1**time)

                p.data -= lr_t * m / (v ** 0.5 + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = time + 1

        return loss

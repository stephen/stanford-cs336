import math
from typing import Callable, Optional
import torch as t

class SGD(t.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> float | None: # type: ignore
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.

        return loss

t.manual_seed(0)
weights = t.nn.Parameter(5 * t.randn((10, 10)))
def run_with_lr(lr: float) -> float:
    opt = SGD([weights], lr=lr)
    loss = t.zeros(0)
    for _ in range(10):
        opt.zero_grad()
        loss = (weights**2).mean()
        loss.backward()
        opt.step() # Run optimizer step.
    return loss.cpu().item()

losses = []
for lr in [1e1, 1e2, 1e3]:
    loss = run_with_lr(lr)
    print(f"loss for {lr=}: {loss}")
    losses.append(loss)

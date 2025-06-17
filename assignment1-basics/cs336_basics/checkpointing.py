import os
import typing
import torch as t

def save_checkpoint(model: t.nn.Module, optimizer: t.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    obj = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "iteration": iteration,
    }

    t.save(obj, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: t.nn.Module, optimizer: t.optim.Optimizer):
    obj = t.load(src)

    if "model" not in obj:
        raise ValueError("expected model key in loaded state")
    model.load_state_dict(obj["model"])

    if "optim" not in obj:
        raise ValueError("expected optim key in loaded state")
    optimizer.load_state_dict(obj["optim"])

    if "iteration" not in obj:
        raise ValueError("expected iteration key in loaded state")
    return obj["iteration"]

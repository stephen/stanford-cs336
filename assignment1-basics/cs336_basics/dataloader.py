from typing import Optional
import torch as t
import numpy as np

def get_batch(input: np.ndarray, batch_size: int, context_length: int, device: Optional[t.device] = None) -> tuple[t.Tensor, t.Tensor]:
    starts = np.random.randint(0, len(input) - context_length, size=batch_size)
    # Note that we cast to t.long here because pytorch doesn't like indexing tensors using
    # other datatypes.
    x = t.stack([t.from_numpy(input[i:i+context_length]) for i in starts]).to(t.long)
    y = t.stack([t.from_numpy(input[i+1:i+context_length+1]) for i in starts]).to(t.long)

    if device:
        x = x.to(device)
        y = y.to(device)

    return x, y

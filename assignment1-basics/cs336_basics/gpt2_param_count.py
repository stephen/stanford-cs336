from typing import Any
import torch as t
import torchinfo

from cs336_basics.transformer import TransformerLM


defaults = {
    "vocab_size": 50257,
    "context_len": 1024,
    "d_ff": 6400,
    "d_model": 1600,
    "n_layers": 48,
    "n_heads": 25,
    # Use the meta device to avoid actually allocating.
    "device": t.device("meta")
}

def summary_for(**args: Any):
    model = TransformerLM(
        **(defaults if args is None else {**defaults, **args})
    )

    torchinfo.summary(model, depth=1)

summary_for(n_layers=12, d_model=768, n_heads=12, d_ff=768*8//3)
summary_for(n_layers=24, d_model=1024, n_heads=16, d_ff=1024*8//3)
summary_for(n_layers=36, d_model=1280, n_heads=20, d_ff=1280*8//3)
summary_for()

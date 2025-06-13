import torch as t
import torchinfo

from cs336_basics.transformer import TransformerLM

model = TransformerLM(
    vocab_size=50257,
    context_len=1024,
    d_ff=6400,
    d_model=1600,
    n_layers=48,
    n_heads=25,
    # Use the meta device to avoid actually allocating.
    device=t.device("meta"),
)

torchinfo.summary(model)

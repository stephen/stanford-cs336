from networkx import general_random_intersection_graph
from tqdm import tqdm
import argparse
import pathlib
import numpy as np
import torch as t
from dataclasses import dataclass, field
from typing import Optional

from cs336_basics.softmax import softmax
from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import save_checkpoint
from cs336_basics.cross_entropy_loss import cross_entropy
from cs336_basics.dataloader import get_batch
from cs336_basics.tokenizer_cls import Tokenizer
from cs336_basics.train_model import ModelArgs
from cs336_basics.transformer import TransformerLM

import readline

default_device = t.device('mps') if t.backends.mps.is_available() else t.device('cuda') if t.cuda.is_available() else t.device('cpu')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="./data/model.pth", help='Which file to use as the model weights')
    parser.add_argument('--tokenizer-state', type=str, default=None, help='Which files to use as the tokenizer serialized state')
    parser.add_argument('--max-tokens', type=int, default=256, help='Max tokens to generate for a response')
    cli_args = parser.parse_args()

    assert cli_args.model, "--model must be specified"
    assert cli_args.tokenizer_state, "--model must be specified"

    model_args = ModelArgs()
    model = TransformerLM(
        context_len=model_args.context_len,
        d_ff=model_args.d_ff,
        d_model=model_args.d_model,
        n_heads=model_args.n_heads,
        n_layers=model_args.n_layers,
        rope_theta=model_args.rope_theta,
        vocab_size=model_args.vocab_size,
        device=default_device,
    )
    with open(cli_args.model, "rb") as f:
        t.load(f, model.state_dict())

    max_tokens = cli_args.max_tokens
    tokenizer = Tokenizer.from_file(cli_args.tokenizer_state)

    eot_encode = tokenizer.encode("<|endoftext|>")
    assert len(eot_encode) == 1, "expected only one token for <|endoftext|>"
    eot = eot_encode[0]

    while True:
        try:
            line = input("> ").strip()
            if not line:
                continue

            generated = 0
            encoded = t.tensor(tokenizer.encode(line)).unsqueeze(0)
            while True:
                logits = model(encoded)
                output = softmax(logits, dim=-1)
                token_id = t.argmax(output, dim=-1)[0][-1].long().item()
                next = tokenizer.decode([int(token_id)])

                if next == eot:
                    break

                encoded = t.concat([encoded, t.tensor([token_id]).unsqueeze(0)], dim=-1)

                if encoded.shape[-1] > model_args.context_len:
                    encoded = encoded[..., -model_args.context_len:]

                generated += 1
                print(next, end="", flush=True)

                if generated >= max_tokens:
                    break

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        print("")

if __name__ == "__main__":
    main()

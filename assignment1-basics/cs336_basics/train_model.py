from tqdm import tqdm
import argparse
import pathlib
import numpy as np
import torch as t
from dataclasses import dataclass, field
from typing import Optional

from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import save_checkpoint
from cs336_basics.cross_entropy_loss import cross_entropy
from cs336_basics.dataloader import get_batch
from cs336_basics.tokenizer_cls import Tokenizer
from cs336_basics.transformer import TransformerLM

default_device = t.device('mps') if t.backends.mps.is_available() else t.device('cuda') if t.cuda.is_available() else t.device('cpu')

@dataclass
class ModelArgs:
    vocab_size: int = 10000
    context_len: int = 256
    n_layers: int = 4
    d_model: int = 512
    n_heads: int = 16
    d_ff: int = 1344
    rope_theta: Optional[float] = 10000

@dataclass
class OptimizerArgs:
    weight_decay: float = 0.01
    betas: tuple[float, float] = (.9, .999)

    max_learning_rate: float = 1e-4
    min_learning_rate: float = 1e-4/10
    warmup_iters: int = 1000
    cosine_cycle_iters: int = 10000

@dataclass
class TrainingArgs:
    training_set_file: pathlib.Path
    validation_set_file: pathlib.Path

    tokenizer_state_file: Optional[pathlib.Path] = None

    steps: int = 10_000
    model_args: ModelArgs = field(default_factory=ModelArgs)

    batch_size: int = 64
    optimizer_args: OptimizerArgs = field(default_factory=OptimizerArgs)

    wandb_group_name: Optional[str] = None
    wandb_run_name: Optional[str] = None

    device: t.device = default_device


class Trainer:
    def __init__(self, args: TrainingArgs):
        self.args = args

    def setup(self):
        args = self.args

        self.tokenizer = Tokenizer.from_file(str(args.tokenizer_state_file))

        self.model = TransformerLM(
            context_len=args.model_args.context_len,
            d_ff=args.model_args.d_ff,
            d_model=args.model_args.d_model,
            n_heads=args.model_args.n_heads,
            n_layers=args.model_args.n_layers,
            rope_theta=args.model_args.rope_theta,
            vocab_size=args.model_args.vocab_size,
            device=args.device,
        )
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.optimizer_args.max_learning_rate)
        # XXX: schedule lr?

        self.training_set = np.load(self.args.training_set_file, mmap_mode='r')
        self.validation_set = np.load(self.args.validation_set_file, mmap_mode='r')

    def teardown(self):
        # XXX: do we need to un-memmap?
        del self.tokenizer
        del self.training_set
        del self.validation_set
        del self.optimizer
        del self.model

    def training_step(self, x: t.Tensor, label: t.Tensor):
        output = self.model(x)
        loss = cross_entropy(output, label)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def train(self):
        for step in tqdm(range(self.args.steps)):
            x, label = get_batch(self.training_set, self.args.batch_size, self.args.model_args.context_len, device=self.args.device)
            loss = self.training_step(x, label)
            if step % 1000 == 0:
                save_checkpoint(self.model, self.optimizer, step, f"./data/model-checkpoint-{step}.pth")
            if step % 100 == 0:
                print(f"step: {step} loss: {loss.cpu().item():.2f}")

        path = f"./data/model.pth"
        t.save(self.model.state_dict(), path)
        print(f"saved to {path=}")

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-set', type=str, default=None, help='Which file to use as the training set')
    parser.add_argument('--validation-set', type=str, default=None, help='Which file to use as the validation set')
    parser.add_argument('--tokenizer-state', type=str, default=None, help='Which files to use as the tokenizer serialized state')
    cli_args = parser.parse_args()

    assert cli_args.training_set, "--training-set must be specified"
    assert cli_args.validation_set, "--validation-set must be specified"
    assert cli_args.tokenizer_state, "--tokenizer-state must be specified"

    args = TrainingArgs(
        training_set_file=pathlib.Path(cli_args.training_set),
        validation_set_file=pathlib.Path(cli_args.validation_set),
        tokenizer_state_file=pathlib.Path(cli_args.tokenizer_state),
    )

    with Trainer(args) as t:
        t.train()

if __name__ == "__main__":
    main()

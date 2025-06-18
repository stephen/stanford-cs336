from tqdm import tqdm
import pathlib
import numpy as np
import torch as t
from dataclasses import asdict, dataclass, field
from typing import Optional

import wandb

from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import save_checkpoint
from cs336_basics.cross_entropy_loss import cross_entropy
from cs336_basics.dataloader import get_batch
from cs336_basics.lr_cosine_schedule import lr_cosine_schedule
from cs336_basics.tokenizer_cls import Tokenizer
from cs336_basics.transformer import TransformerLM
from cs336_basics.gradient_clipping import clip_gradients


default_device = t.device('mps:0') if t.backends.mps.is_available() else t.device('cuda') if t.cuda.is_available() else t.device('cpu')
default_backend = "aot_eager" if default_device.type == "mps" else "inductor"

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

    max_learning_rate: float = 3e-4
    min_learning_rate: float =   1e-5
    warmup_iters: int = 500
    cosine_cycle_iters: int = 4000

@dataclass
class TrainingArgs:
    training_set: pathlib.Path
    validation_set: pathlib.Path

    tokenizer_state: pathlib.Path

    validation_step_interval: int = 100
    checkpoint_step_interval: int = 1000

    model_args: ModelArgs = field(default_factory=ModelArgs)

    steps: int = 5000
    batch_size: int = 32

    optimizer_args: OptimizerArgs = field(default_factory=OptimizerArgs)
    clip_gradient_to_max_norm: Optional[float] = 1.0

    wandb_group_name: Optional[str] = None
    wandb_run_name: Optional[str] = None

    device: t.device = default_device


class Trainer:
    def __init__(self, args: TrainingArgs):
        self.args = args

    def setup(self):
        args = self.args

        self.tokenizer = Tokenizer.from_file(str(args.tokenizer_state))

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
        self.model.compile(backend=default_backend)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.optimizer_args.max_learning_rate,
            weight_decay=self.args.optimizer_args.weight_decay,
            betas=self.args.optimizer_args.betas,
        )

        self.training_set = np.load(self.args.training_set, mmap_mode='r')
        self.validation_set = np.load(self.args.validation_set, mmap_mode='r')
        wandb.init(
            project="cs336-llm",
            config=asdict(self.args),
            group=self.args.wandb_group_name,
            name=self.args.wandb_run_name,
        )
        wandb.watch(self.model, log="gradients", log_freq=10)

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

        if self.args.clip_gradient_to_max_norm is not None:
            clip_gradients(
                self.model.parameters(),
                self.args.clip_gradient_to_max_norm,
            )

        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def evaluate(self):
        self.model.eval()
        x, label = get_batch(self.validation_set, self.args.batch_size, self.args.model_args.context_len, device=self.args.device)
        output = self.model(x)
        loss = cross_entropy(output, label)
        perplexity = loss.exp()

        return loss, perplexity

    def lr_for_step(self, step: int):
        return lr_cosine_schedule(
            step,
            self.args.optimizer_args.max_learning_rate,
            self.args.optimizer_args.min_learning_rate,
            self.args.optimizer_args.warmup_iters,
            self.args.optimizer_args.cosine_cycle_iters
        )

    def train(self):
        iter = tqdm(range(self.args.steps))
        valid_loss, valid_perplexity = self.evaluate()
        wandb.log({"valid_loss": valid_loss, "valid_perplexity": valid_perplexity}, step=0)

        for step in iter:
            self.model.train()

            lr = self.lr_for_step(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            x, label = get_batch(self.training_set, self.args.batch_size, self.args.model_args.context_len, device=self.args.device)
            test_loss = self.training_step(x, label)

            if step % self.args.checkpoint_step_interval == 0:
                save_checkpoint(self.model, self.optimizer, step, f"./data/model-checkpoint-{step}.pth")

            if step % self.args.validation_step_interval == 0:
                valid_loss, valid_perplexity = self.evaluate()
                iter.set_postfix({})

            iter.set_postfix({
                "train_loss": f"{test_loss.cpu().item():.2f}",
                "valid_loss": f"{valid_loss.cpu().item():.2f}",
                "valid_perplexity": f"{valid_perplexity.cpu().item():.2f}",
            })
            wandb.log({
                "test_loss": test_loss,
                "valid_loss": valid_loss,
                "valid_perplexity": valid_perplexity,
                "lr": lr,
            }, step=step)

        path = f"./data/model.pth"
        t.save(self.model.state_dict(), path)
        print(f"saved to {path=}")

        artifact = wandb.Artifact(path, type="model")
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        wandb.finish()

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()
        return False


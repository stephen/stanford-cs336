from functools import partial
from tqdm import tqdm
import pathlib
import numpy as np
import torch as t
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, cast

import os
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import save_checkpoint
from cs336_basics.cross_entropy_loss import cross_entropy
from cs336_basics.dataloader import get_batch
from cs336_basics.lr_cosine_schedule import lr_cosine_schedule
from cs336_basics.tokenizer_cls import Tokenizer
from cs336_basics.transformer import TransformerLM
from cs336_basics.gradient_clipping import clip_gradients
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module, loss_parallel
from torch.distributed.tensor import distribute_tensor, DTensor



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

    local_rank: int = -1
    world_size: int = -1

    validation_step_interval: Optional[int] = 100
    checkpoint_step_interval: Optional[int] = 1000

    compile: bool = False
    model_args: ModelArgs = field(default_factory=ModelArgs)

    steps: int = 2000
    batch_size: int = 24

    optimizer_args: OptimizerArgs = field(default_factory=OptimizerArgs)
    clip_gradient_to_max_norm: Optional[float] = 1.0

    wandb_group_name: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_log: Optional[str] = None

    device: t.device = default_device

    # Parallelisms.
    dp: int = -1
    tp: int = -1
    loss_parallel: bool = False
    fsdp: int = -1


class DistributedTrainer:
    def __init__(self, args: TrainingArgs, mesh: Any):
        self.args = args
        self.mesh = mesh

    def setup(self):
        args = self.args

        self.tokenizer = Tokenizer.from_file(str(args.tokenizer_state))

        if args.dp > 0 or args.fsdp > 0:
            if not dist.is_initialized():
                dist.init_process_group("nccl") # DDP() is old, so we still need to init process group for it
                t.cuda.set_device(self.args.local_rank)

        self.model = TransformerLM(
            context_len=args.model_args.context_len,
            d_ff=args.model_args.d_ff,
            d_model=args.model_args.d_model,
            n_heads=args.model_args.n_heads,
            n_layers=args.model_args.n_layers,
            rope_theta=args.model_args.rope_theta,
            vocab_size=args.model_args.vocab_size,
            device=args.device,
            mesh=self.mesh,
            tp=self.args.tp,
            loss_parallel=args.loss_parallel,
        )

        if args.dp > 0:
            self.model = DDP(self.model, device_ids=[self.args.local_rank])

        def backward_hook(name, module, grad_output):
            if dist.get_rank() != 0:
                return
            print(f"[{dist.get_rank()}] Backward through {module.__class__.__name__} @ ---{name}---")
            for i, g in enumerate(grad_output):
                if g is not None:
                    print(f"[{dist.get_rank()}]    grad_output[{i}]: DTensor={isinstance(g, DTensor)}, shape={g.shape}")
            # for i, g in enumerate(grad_input):
            #     if g is not None:
            #         print(f"[{dist.get_rank()}]    grad_input[{i}]: DTensor={isinstance(g, DTensor)}, shape={g.shape}")


            return None

        def backward_hook_complete(name, module, grad_input, grad_output):
            if dist.get_rank() != 0:
                return
            print(f"[{dist.get_rank()}] Backward through {module.__class__.__name__} @ ---{name}--- DONE")
            return None



        # Register on all modules
        # for name, module in self.model.named_modules():
        #     module.register_full_backward_pre_hook(partial(backward_hook, name))
        #     module.register_full_backward_hook(partial(backward_hook_complete, name))

        # When logging parameters, compile doesn't play well.
        if self.args.compile and self.args.wandb_log == "gradients":
            self.model.compile(backend=default_backend)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.optimizer_args.max_learning_rate,
            weight_decay=self.args.optimizer_args.weight_decay,
            betas=self.args.optimizer_args.betas,
        )

        # self.training_set = np.load(self.args.training_set, mmap_mode='r')
        # self.validation_set = np.load(self.args.validation_set, mmap_mode='r')
        self.training_set = np.load(self.args.training_set)
        self.validation_set = np.load(self.args.validation_set)

        if self.args.local_rank == 0:
            wandb.init(
                project="timlm",
                config=asdict(self.args),
                group=self.args.wandb_group_name,
                name=self.args.wandb_run_name,
            )
            wandb.watch(self.model, log=cast(Literal["gradients", "parameters", "all"], self.args.wandb_log), log_freq=10)

    def teardown(self):
        dist.destroy_process_group()
        del self.tokenizer
        del self.training_set
        del self.validation_set
        del self.optimizer
        del self.model

    def training_step(self, x: t.Tensor, label: t.Tensor):
        output = self.model(x)
        if self.args.loss_parallel:
            with loss_parallel():
                loss = t.nn.functional.cross_entropy(output.flatten(0, 1), label.flatten(0, 1))
                loss.backward()
        else:
            if isinstance(label, DTensor):
                label = label.to_local()
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
        return t.tensor(0.0), t.tensor(0.0)
        self.model.eval()
        if self.args.local_rank == 0:
            x, label = get_batch(self.validation_set, self.args.batch_size, self.args.model_args.context_len, device=self.args.device)
        else:
            x = t.empty((self.args.batch_size, self.args.model_args.context_len), device=self.args.device).long()
            label = t.empty((self.args.batch_size, self.args.model_args.context_len), device=self.args.device).long()

        if self.args.tp > 1:
            x = distribute_tensor(x, device_mesh=self.mesh)
            label = distribute_tensor(label, device_mesh=self.mesh).to_local()

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
        # t.autograd.set_detect_anomaly(True)
        iter = tqdm(range(self.args.steps))
        valid_loss, valid_perplexity = self.evaluate()
        if self.args.local_rank == 0:
            wandb.log({"valid_loss": valid_loss, "valid_perplexity": valid_perplexity}, step=0)

        for step in iter:
            self.model.train()

            lr = self.lr_for_step(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # print("local rank", self.args.local_rank)
            if self.args.local_rank == 0:
                x, label = get_batch(self.training_set, self.args.batch_size, self.args.model_args.context_len, device=self.args.device)
            else:
                x = t.empty((self.args.batch_size, self.args.model_args.context_len), device=self.args.device).long()
                label = t.empty((self.args.batch_size, self.args.model_args.context_len), device=self.args.device).long()

            if self.args.tp > 1:
                x = distribute_tensor(x, device_mesh=self.mesh)
                label = distribute_tensor(label, device_mesh=self.mesh)

            test_loss = self.training_step(x, label)

            if self.args.checkpoint_step_interval is not None and step % self.args.checkpoint_step_interval == 0:
                save_checkpoint(self.model, self.optimizer, step, f"./data/model-checkpoint-{step}.pth")

            if self.args.validation_step_interval is not None and step % self.args.validation_step_interval == 0:
                valid_loss, valid_perplexity = self.evaluate()
                iter.set_postfix({})

            iter.set_postfix({
                "train_loss": f"{test_loss.cpu().item():.2f}",
                "valid_loss": f"{valid_loss.cpu().item():.2f}",
                "valid_perplexity": f"{valid_perplexity.cpu().item():.2f}",
            })
            if self.args.local_rank == 0:
                wandb.log({
                    "test_loss": test_loss,
                    "valid_loss": valid_loss,
                    "valid_perplexity": valid_perplexity,
                    "lr": lr,
                }, step=step)

        path = f"./data/model.pth"
        t.save(self.model.state_dict(), path)
        print(f"saved to {path=}")

        if wandb.run and wandb.run.sweep_id is None:
            artifact = wandb.Artifact(pathlib.Path(path).name, type="model")
            artifact.add_file(path)
            wandb.log_artifact(artifact)
            wandb.finish()

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()
        return False


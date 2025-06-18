import wandb
import simple_parsing

from cs336_basics.trainer import Trainer, TrainingArgs

sweep_configuration = {
    "name": "tinystories-hp",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "valid_loss"},
    "parameters": {
        "learning_rate": {"min": 1e-5, "max": 1e-3},
        "learning_rate_warmup": {"min": 0, "max": 2000},
        "learning_rate_cosine": {"min": 0, "max": 4000},
        "batch_size": {"values": [16, 32, 64, 128]},
        "weight_decay": {"min": 0, "max": 1},
        "beta1": {"min": .9, "max": .99},
        "beta2": {"min": .95, "max": .999},
        "clip_gradient_to_max_norm": {"min": 0.5, "max": 1.5}
    },
}

parser = simple_parsing.ArgumentParser()
parser.add_arguments(TrainingArgs, dest="parsed")
cli_args = parser.parse_args()

def train():
    wandb.init(project="cs336-llm")
    # wandb.init(project="cs336-llm", reinit=True)

    # Define args & initialize wandb
    args: TrainingArgs = cli_args.parsed

    args.steps = 500
    args.checkpoint_step_interval = None
    args.validation_step_interval = 10

    args.optimizer_args.max_learning_rate = wandb.config["learning_rate"]
    args.optimizer_args.min_learning_rate = args.optimizer_args.max_learning_rate / 10
    args.optimizer_args.cosine_cycle_iters = wandb.config["learning_rate_cosine"]
    args.optimizer_args.warmup_iters = wandb.config["learning_rate_warmup"]
    args.batch_size = wandb.config["batch_size"]
    args.optimizer_args.weight_decay = wandb.config["weight_decay"]
    args.optimizer_args.betas = (wandb.config["beta1"], wandb.config["beta2"])
    args.clip_gradient_to_max_norm = wandb.config["clip_gradient_to_max_norm"]

    with Trainer(args) as t:
        t.train()

def main():
    sweep_id = wandb.sweep(sweep_configuration, project="cs336-llm")
    wandb.agent(sweep_id=sweep_id, function=train)

if __name__ == "__main__":
    main()

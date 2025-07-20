import os
import simple_parsing
import torch.multiprocessing as mp

from cs336_basics.trainer_distributed import DistributedTrainer, TrainingArgs

def main():
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(TrainingArgs, dest="parsed")
    cli_args = parser.parse_args()

    training_args: TrainingArgs
    training_args = cli_args.parsed

    training_args.local_rank = int(os.environ["LOCAL_RANK"])
    training_args.world_size = int(os.environ["WORLD_SIZE"])

    with DistributedTrainer(training_args) as t:
        t.train()

if __name__ == "__main__":
    world_size = 2
    main()

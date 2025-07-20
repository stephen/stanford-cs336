import simple_parsing
import torch.multiprocessing as mp

from cs336_basics.trainer_distributed import DistributedTrainer, TrainingArgs

def main(rank: int, world_size: int):
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(TrainingArgs, dest="parsed")
    cli_args = parser.parse_args()

    training_args: TrainingArgs
    training_args = cli_args.parsed
    training_args.local_rank = rank
    training_args.world_size = world_size

    with DistributedTrainer(training_args) as t:
        t.train()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(main,
             args=(world_size,),
             nprocs=world_size,
             join=True)

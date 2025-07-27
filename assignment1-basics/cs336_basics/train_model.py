import os
import simple_parsing
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
import numpy as np
import torch as t

from torch.distributed.tensor.debug import CommDebugMode

from cs336_basics.trainer_distributed import DistributedTrainer, TrainingArgs

def main():
    np.random.seed(5)
    t.manual_seed(5)

    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(TrainingArgs, dest="parsed")
    cli_args = parser.parse_args()
    training_args: TrainingArgs
    training_args = cli_args.parsed

    if training_args.tp > 1:
        mesh = init_device_mesh("cuda", (training_args.tp,), mesh_dim_names=("tp", ),)
    else:
        mesh = None

    training_args.local_rank = int(os.environ["LOCAL_RANK"])
    training_args.world_size = int(os.environ["WORLD_SIZE"])

    with DistributedTrainer(training_args, mesh) as trainer:
        # with CommDebugMode():
        trainer.train()

if __name__ == "__main__":
    world_size = 2
    main()

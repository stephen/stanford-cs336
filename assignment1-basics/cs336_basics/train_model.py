import simple_parsing

from cs336_basics.trainer import Trainer, TrainingArgs

def main():
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(TrainingArgs, dest="parsed")
    cli_args = parser.parse_args()


    with Trainer(cli_args.parsed) as t:
        t.train()

if __name__ == "__main__":
    main()

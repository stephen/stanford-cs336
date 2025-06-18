from tqdm import tqdm
import argparse
import pathlib
import pickle
import numpy as np

from cs336_basics.tokenizer_cls import Tokenizer
from cs336_basics.parallel_pretokenizer import parallel_pretokenize_path_to_corpus
from cs336_basics.tokenizer import bpe_tokenize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="./data/TinyStoriesV2-GPT4-train.txt", help='which file to use as the input')
    parser.add_argument('--vocab-size', type=int, default=10000, help='vocab size to merge')
    parser.add_argument('--tokenizer-state', type=str, default=None, help='a pre-existing tokenizer state to use')
    cli_args = parser.parse_args()

    path = pathlib.Path(cli_args.input)

    if not cli_args.tokenizer_state:
        print(f"training tokenizer: {path}")
        corpus = parallel_pretokenize_path_to_corpus(path)

        special_tokens = ["<|endoftext|>"]
        vocab, merges = bpe_tokenize(corpus, cli_args.vocab_size, special_tokens)

        t = Tokenizer(vocab, merges, special_tokens)

        outpath = path.with_suffix(".tokenizer_pkl")
        t.to_file(outpath)
        print(f"tokenizer state saved to: {outpath}")
    else:
        state_path = cli_args.tokenizer_state
        t = Tokenizer.from_file(state_path)
        print(f"using tokenizer state from: {state_path}")

    print(f"tokenizing: {path}")
    output: list[int]
    with open(path, "r") as f:
        tokens = t.encode_iterable(f)
        output = list(tqdm(tokens))

    outpath = path.with_suffix(".tokenized_pkl")
    with open(outpath, "wb") as f:
        pickle.dump({
            "tokens": np.array(output, dtype=np.uint16),
        }, f)
    print(f"tokenized text saved to: {outpath}")

if __name__ == "__main__":
    main()

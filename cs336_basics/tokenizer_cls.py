import os
from cs336_basics.parallel_pretokenizer import parallel_pretokenize_path_to_corpus
from cs336_basics.tokenizer import Vocab, bpe_tokenize


Merges = list[tuple[bytes, bytes]]

def train_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[Vocab, Merges]:
    corpus = parallel_pretokenize_path_to_corpus(input_path)

    merges = vocab_size - 256
    assert merges > 0, "vocab_size cannot be below the base 256 tokens"

    return bpe_tokenize(corpus, merges, special_tokens)

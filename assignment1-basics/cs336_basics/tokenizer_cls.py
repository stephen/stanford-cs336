import os
import pickle
from typing import Iterable
from cs336_basics.parallel_pretokenizer import parallel_pretokenize_path_to_corpus
from cs336_basics.tokenizer import Vocab, bpe_tokenize, Merges, merge_at_positions, pretokenize_to_corpus

def train_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[Vocab, Merges]:
    corpus = parallel_pretokenize_path_to_corpus(input_path, special_tokens)

    merges = vocab_size - 256 - len(special_tokens)
    assert merges > 0, "vocab_size cannot be below the base 256 tokens"

    return bpe_tokenize(corpus, merges, special_tokens)

class Tokenizer:
    def __init__(
        self,
        vocab: Vocab,
        merges: Merges,
        special_tokens: list[str] = [],
    ):
        self.vocab = vocab
        self.reverse_vocab = {seq: id for id, seq in self.vocab.items()}
        self.merges = merges
        self.special_tokens = [t.encode('utf-8') for t in special_tokens]

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] =[],
    ):
        with open(vocab_filepath, "rb") as f:
            v = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            m = pickle.load(f)
        return cls(v, m, special_tokens)

    def encode(self, text: str) -> list[int]:
        return [id for id in self.encode_iterable([text])]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for chunk in iterable:
            pretokens = pretokenize_to_corpus(chunk.encode('utf-8'), self.special_tokens)
            rv = []
            for pretoken in pretokens:
                p = pretoken
                merges_i = 0
                # For each pretoken, we only need to do one pass of merges because earlier
                # merges could not have happened after later ones.
                while merges_i < len(self.merges):
                    i = 0
                    m = self.merges[merges_i]

                    # We may need to try a given merge several times in case it recurs in the same pretoken.
                    while True:
                        try_again = False
                        for j, pair in enumerate(zip(p[i:], p[i+1:])):
                            if m == pair:
                                p = merge_at_positions(p, m, [i + j])
                                i = j +1
                                try_again = True
                                break
                        if not try_again:
                            break
                    merges_i += 1

                for token in p:
                    yield self.reverse_vocab[token]



    def decode(self, ids: list[int]) -> str:
        pass

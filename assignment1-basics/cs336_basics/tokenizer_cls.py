import regex as re
import os
import pickle
import struct
from typing import Iterable, Optional
from cs336_basics.parallel_pretokenizer import parallel_pretokenize_path_to_corpus
from cs336_basics.tokenizer import Vocab, bpe_tokenize, Merges, merge_at_positions, pretokenize

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
        special_tokens: Optional[list[str]] = None,
    ):
        self.vocab = vocab
        self.reverse_vocab = {seq: id for id, seq in self.vocab.items()}
        self.merges = merges
        self.special_tokens = sorted([t.encode('utf-8') for t in (special_tokens or [])], reverse=True)
        self.merge_lookup = {merge: i for i, merge in enumerate(self.merges)}

    @classmethod
    def from_file(
        cls,
        filepath: str | os.PathLike,
    ):
        with open(filepath, "rb") as f:
            d = pickle.load(f)
            return cls(d["vocab"], d["merges"], d["special_tokens"])

    def to_file(
        self,
        filepath: str | os.PathLike,
    ):
        with open(filepath, "wb") as f:
            pickle.dump({
                "vocab": self.vocab,
                "merges": self.merges,
                "special_tokens": [t.decode('utf-8') for t in self.special_tokens],
            }, f)

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

    def to_files(
        self,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
    ):
        with open(vocab_filepath, "wb") as f:
            pickle.dump(self.vocab, f)
        with open(merges_filepath, "wb") as f:
            pickle.dump(self.merges, f)

    def encode(self, text: str) -> list[int]:
        return [id for id in self.encode_iterable([text])]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for chunk in iterable:
            pretokens = pretokenize(chunk.encode('utf-8'), self.special_tokens)
            for pretoken in pretokens:
                maybe_special_token = b''.join(pretoken)
                if maybe_special_token in self.special_tokens:
                    yield self.reverse_vocab[maybe_special_token]
                    continue

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
        return b''.join([self.vocab[id] for id in ids]).decode('utf-8', errors='replace')

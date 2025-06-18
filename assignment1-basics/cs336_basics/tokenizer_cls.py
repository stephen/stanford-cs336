import regex as re
import os
import pickle
import struct
from typing import Iterable, Optional, Sequence
from cs336_basics.parallel_pretokenizer import parallel_pretokenize_path_to_corpus
from cs336_basics.tokenizer import Token, Vocab, bpe_tokenize, Merges, merge_at_positions, pretokenize

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
        pretoken_cache: dict[Sequence, Sequence] = {}
        for chunk in iterable:
            pretokens = pretokenize(chunk.encode('utf-8'), self.special_tokens)
            for pretoken in pretokens:
                maybe_special_token = b''.join(pretoken)
                if maybe_special_token in self.special_tokens:
                    yield self.reverse_vocab[maybe_special_token]
                    continue

                p = pretoken

                if p in pretoken_cache:
                    for token in pretoken_cache[p]:
                        yield self.reverse_vocab[token]
                    continue

                while True:
                    # Locate the next best merge, defined as being the earliest merge
                    # in our merges index.
                    earliest_merge: Optional[tuple[int, int]] = None # tuple of pretoken index and merge index
                    for i, pair in enumerate(zip(p, p[1:])):
                        merge_i = self.merge_lookup.get(pair, -1)
                        if merge_i != -1 and (not earliest_merge or merge_i < earliest_merge[1]):
                            earliest_merge = (i, merge_i)

                    if earliest_merge is not None:
                        i, _ = earliest_merge
                        pair = (p[i], p[i+1])
                        p = merge_at_positions(p, pair, [i])
                    else:
                        break

                pretoken_cache[pretoken] = p
                for token in p:
                    yield self.reverse_vocab[token]

    def decode(self, ids: list[int]) -> str:
        return b''.join([self.vocab[id] for id in ids]).decode('utf-8', errors='replace')

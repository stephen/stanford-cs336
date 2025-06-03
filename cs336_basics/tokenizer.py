from collections import defaultdict
from typing import TypedDict
import regex as re

PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize(input: str):
    return re.finditer(PAT, input)

# _Refs is a dict of the token tuple -> the position of the pair (first index)
_Refs = dict[tuple[bytes, ...], int]


def bpe_tokenize(text: str, num_merges: int) -> list[bytes]:
    vocab = list([b.to_bytes() for b in range(0, 256)])
    vocab.append('<|endoftext|>'.encode('utf-8'))

    # dict of token tuple -> count
    corpus: dict[tuple[bytes, ...], int] = defaultdict(int)

    for pretoken in pretokenize(text):
        corpus[tuple(bytes([b]) for b in pretoken.group().encode('utf-8'))] += 1

    for _ in range(num_merges):
        pairs: dict[tuple[bytes, bytes], _Refs] = {}
        for pretoken, count in corpus.items():
            for i, (a, b) in enumerate(zip(pretoken, pretoken[1:])):
                key = (a, b)
                if key not in pairs:
                    # XXX: can we intern this?
                    pairs[key] = {}
                pairs[key][pretoken] = i

        # max_pair = max(pairs, key=lambda k: (pairs[k]["count"], k))
        max_pair = max(pairs, key=lambda k: (sum(corpus[k] for k in pairs[k].keys()), k))
        vocab.append(b''.join(max_pair))

        for old_key, i in pairs[max_pair].items():
            key = old_key[:i] + (max_pair[0] + max_pair[1],) + old_key[i+2:]
            corpus[key] = corpus[old_key]
            del corpus[old_key]
            # for all of the ones we touched, we should go back and re-count the tokens in it

    return vocab

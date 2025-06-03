from collections import defaultdict
from typing import TypedDict
import regex as re

PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize(input: str):
    return re.finditer(PAT, input)

class _PairCount(TypedDict):

    count: int
    # refs is where this pair occurred. the ref is a pretoken tuple and the position of the pair.
    refs: list[tuple[tuple[bytes, ...], int]]


def bpe_tokenize(text: str, num_merges: int) -> list[bytes]:
    vocab = list([b.to_bytes() for b in range(0, 256)])
    vocab.append('<|endoftext|>'.encode('utf-8'))

    corpus: dict[tuple[bytes, ...], int] = defaultdict(int)

    for pretoken in pretokenize(text):
        corpus[tuple(bytes([b]) for b in pretoken.group().encode('utf-8'))] += 1

    for _ in range(num_merges):
        pairs: dict[tuple[bytes, bytes], _PairCount] = {}
        for pretoken, count in corpus.items():
            for i, (a, b) in enumerate(zip(pretoken, pretoken[1:])):
                key = (a, b)
                if key not in pairs:
                    # XXX: can we intern this?
                    pairs[key] = {'count': 0, 'refs': []}
                pairs[key]['count'] += count
                pairs[key]['refs'].append((pretoken, i))

        max_pair = max(pairs, key=lambda k: (pairs[k]["count"], k))
        vocab.append(b''.join(max_pair))

        for old_key, i in pairs[max_pair]['refs']:
            key = old_key[:i] + (max_pair[0] + max_pair[1],) + old_key[i+2:]
            corpus[key] = corpus[old_key]
            del corpus[old_key]

    return vocab

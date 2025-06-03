from collections import defaultdict
import struct
import time
from typing import TypedDict
import regex as re

PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""".encode('utf-8')

def pretokenize(input: bytes):
    return re.finditer(PAT, input)

# _Token is a token. It may be one or more bytes.
_Token = bytes

# _Sequence is an arbitrary sequence of tokens.
_Sequence = tuple[_Token, ...]

# _Refs is a dict of sequence -> position in sequence for a token pair.
_Refs = dict[_Sequence, int]

# _TokenPair is a pair of tokens.
_TokenPair = tuple[_Token, _Token]

class _TokenPairCounts(TypedDict):
    count: int
    refs: _Refs

def bpe_tokenize(text: bytes, num_merges: int) -> list[bytes]:
    vocab = list([b.to_bytes() for b in range(0, 256)])
    vocab.append('<|endoftext|>'.encode('utf-8'))

    # Corpus is the text of sequences to the count of times they happen.
    corpus: dict[_Sequence, int] = defaultdict(int)
    for pretoken in pretokenize(text): # text is bytes, returns regex.finditer's result for bytes
        g = pretoken.group()

        # fast
        key = struct.unpack('c' * len(g), g)

        # medium slow
        # key = tuple([g[i:i+1] for i in range(len(g))])

        # slow
        # mv = memoryview(g)
        # key = tuple(mv[i:i+1] for i in range(len(mv)))

        # slowest
        # key = tuple(bytes([b]) for b in pretoken.group().encode('utf-8'))

        corpus[key] += 1

    # pairs is the state of all token pairs in the corpus, pointing to where they happen in the corpus and the total count the pair happens.
    pairs: dict[_TokenPair, _TokenPairCounts] = {}

    # Go through each sequence that needs to be checked. At the start, this is the entire corpus but
    # later this is only sequences that need to be rechecked.
    corpus_to_check = corpus.keys()
    for _ in range(num_merges):
        for pretoken in corpus_to_check:
            count = corpus[pretoken]
            for i, pair in enumerate(zip(pretoken, pretoken[1:])):
                if pair not in pairs:
                    pairs[pair] = {'count': 0, 'refs': {}}
                pairs[pair]['refs'][pretoken] = i
                pairs[pair]['count'] += count

        # Figure out the max pair.
        # XXX: We could maybe keep track of this state during the loop.
        max_pair = max(pairs, key=lambda k: (pairs[k]['count'], k))
        vocab.append(b''.join(max_pair))

        # Invalidate pairs only at the spots where the sequence changed because the underlying tokens
        # changed due to max_pair.
        corpus_to_check = set()
        pairs_to_invalidate: set[tuple[_TokenPair, _Sequence, _Sequence]] = set()
        for old_seq, i in pairs[max_pair]['refs'].items():
            new_seq = old_seq[:i] + (max_pair[0] + max_pair[1],) + old_seq[i+2:]

            # For all pairs of the old sequence, remove their reference.
            for i, neighbor_pair in enumerate(zip(old_seq, old_seq[1:])):
                pairs_to_invalidate.add((neighbor_pair, old_seq, new_seq))

            corpus_to_check.add(new_seq)
            corpus[new_seq] = corpus[old_seq]
            del corpus[old_seq]

        for (neighbor_pair, old_seq, new_seq) in pairs_to_invalidate:
            del pairs[neighbor_pair]['refs'][old_seq]
            pairs[neighbor_pair]['count'] -= corpus[new_seq]

    return vocab

with open("./data/TinyStoriesV2-GPT4-train.txt", "rb") as f:
    content = f.read()
    start = time.perf_counter()
    bpe_tokenize(content, 1000)
    print("time", time.perf_counter() - start)

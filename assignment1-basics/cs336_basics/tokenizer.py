from collections import defaultdict
import struct
from typing import Iterator, Optional, TypedDict
import regex as re

PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""".encode('utf-8')

# Token is a token. It may be one or more bytes.
Token = bytes

# Sequence is an arbitrary sequence of tokens.
Sequence = tuple[Token, ...]

# _Refs is a dict of sequence -> positions in sequence for a token pair. A sequence may contain a pair
# multiple times.
_Refs = dict[Sequence, list[int]]

# TokenPair is a pair of tokens.
TokenPair = tuple[Token, Token]

# Corpus is the input pretokenized corpus with frequencies.
Corpus = dict[Sequence, int]

# Vocab is the output dictionary of ID (index) -> bytes.
Vocab = dict[int, bytes]

# Merges is a list of merges that were done, in order.
Merges = list[TokenPair]

class _TokenPairCounts(TypedDict):
    count: int
    refs: _Refs

def merge_corpora(*corpora: Corpus) -> Corpus:
    rv: Corpus = defaultdict(int)
    for c in corpora:
        for seq, count in c.items():
            rv[seq] += count
    return rv


def pretokenize(text: bytes, special_tokens: Optional[list[bytes]] = None) -> Iterator[Sequence]:
    if special_tokens:
        pattern: bytes = b'(' + b'|'.join([re.escape(token) for token in special_tokens]) + b')'
        splits = re.splititer(pattern, text)
    else:
        splits = [text]

    for chunk in splits:
        if special_tokens and chunk in special_tokens:
            g = chunk
            yield struct.unpack('c' * len(g), g)
            continue
        for pretoken in re.finditer(PAT, chunk):
            g = pretoken.group()
            yield struct.unpack('c' * len(g), g)

def pretokenize_to_corpus(text: bytes, special_tokens: Optional[list[bytes]] = None) -> Corpus:
    corpus: Corpus = defaultdict(int)
    for pretoken in pretokenize(text, special_tokens):
        corpus[pretoken] += 1

    return corpus

def decode_seq(seq: Sequence) -> str:
    def d(b: bytes) -> str:
        try:
            return b.decode('utf-8')
        except:
            return str(b)

    return "'" + ','.join(d(b) for b in seq) + "'"

def hex_seq(seq: Sequence) -> str:
    return f"{''.join(['{:>8}'.format(b.hex()) for b in seq])}"

# merge_at_positions expects positions to be in sorted order.
def merge_at_positions(old_seq: Sequence, merged_pair: TokenPair, positions: list[int]) -> Sequence:
    new_seq = old_seq
    adjust = 0
    for p in positions:
        i = p - adjust
        if new_seq[i:i+2] != merged_pair:
            continue
        new_seq = new_seq[:i] + (merged_pair[0] + merged_pair[1],) + new_seq[i+2:]
        adjust += 1

    return new_seq

def bpe_tokenize(corpus: Corpus, num_merges: int, special_tokens: list[str] = ["<|endoftext|>"]) -> tuple[Vocab, Merges]:
    vocab = list([b.to_bytes() for b in range(0, 256)])
    for t in special_tokens:
        vocab.append(t.encode('utf-8'))

    merges: list[TokenPair] = []

    # pairs is the state of all token pairs in the corpus, pointing to where they happen in the corpus and the total count the pair happens.
    pairs: dict[TokenPair, _TokenPairCounts] = {}

    # Go through each sequence that needs to be checked. At the start, this is the entire corpus but
    # later this is only sequences that need to be rechecked.
    corpus_to_check = corpus.keys()
    for _ in range(num_merges):
        for pretoken in corpus_to_check:
            count = corpus[pretoken]
            for i, pair in enumerate(zip(pretoken, pretoken[1:])):
                if pair not in pairs:
                    pairs[pair] = {'count': 0, 'refs': {}}
                if pretoken not in pairs[pair]['refs']:
                    pairs[pair]['refs'][pretoken] = []
                pairs[pair]['refs'][pretoken].append(i)
                pairs[pair]['count'] += count


        # debug assert: verify that we counted correctly.
        # for p, c in pairs.items():
        #     expected = sum([corpus[ref] * len(pos) for ref, pos in c['refs'].items()])
        #     assert c['count'] == expected, f"{p=}, expected: {expected}, got: {c['count']}"

        # Figure out the max pair.
        max_pair = max(pairs, key=lambda k: (pairs[k]['count'], k))
        vocab.append(b''.join(max_pair))
        merges.append(max_pair)

        # Invalidate pairs only at the spots where the sequence changed because the underlying tokens
        # changed due to max_pair.
        corpus_to_check = set()
        pairs_to_invalidate: set[tuple[TokenPair, Sequence, Sequence]] = set()
        for old_seq, positions in pairs[max_pair]['refs'].items():
            new_seq = merge_at_positions(old_seq, max_pair, positions)

            # For all pairs of the old sequence, remove their reference.
            for i, neighbor_pair in enumerate(zip(old_seq, old_seq[1:])):
                pairs_to_invalidate.add((neighbor_pair, old_seq, new_seq))

            corpus_to_check.add(new_seq)
            corpus[new_seq] = corpus[old_seq]
            del corpus[old_seq]

        for (neighbor_pair, old_seq, new_seq) in pairs_to_invalidate:
            occurences = pairs[neighbor_pair]['refs'][old_seq] # The token might appear multiple times.
            del pairs[neighbor_pair]['refs'][old_seq]

            pairs[neighbor_pair]['count'] -= corpus[new_seq] * len(occurences)

        del pairs[max_pair]

    return {i: v for i, v in enumerate(vocab)}, merges

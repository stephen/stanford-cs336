from cs336_basics.tokenizer_cls import Tokenizer
from .tokenizer import bpe_tokenize, pretokenize_to_corpus

def test_basic():
  text = b"low low low low low\nlower lower widest widest widest\nnewest newest newest newest newest newest"
  corpus = pretokenize_to_corpus(text)
  tokens, merges = bpe_tokenize(corpus, 6, ["<|endoftext|>"])
  assert len(tokens) == 263
  assert len(merges) == 6
  assert all(t in tokens.values() for t in [b'st', b'est', b'ow', b'low', b'west', b'ne'])

def test_encode():
  text = b"low low low low low\nlower lower widest widest widest\nnewest newest newest newest newest newest"
  corpus = pretokenize_to_corpus(text)
  vocab, merges = bpe_tokenize(corpus, 6, ["<|endoftext|>"])

  t = Tokenizer(vocab, merges, ["<|endoftext|>"])
  encoded = t.encode("newest")
  assert encoded == [262, 261]
  assert [vocab[v] for v in encoded] == [b'ne', b'west']

def test_encode_recurrence():
  text = b"construction onion"
  corpus = pretokenize_to_corpus(text)
  vocab, merges = bpe_tokenize(corpus, 1, ["<|endoftext|>"])

  t = Tokenizer(vocab, merges, ["<|endoftext|>"])
  encoded = t.encode("construction")
  assert len(encoded) == len("construction") - 2
  assert encoded[1] == encoded[9]

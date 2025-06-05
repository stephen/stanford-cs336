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

def test_iterable():
  text = b"low low low low low\nlower lower widest widest widest\nnewest newest newest newest newest newest"
  corpus = pretokenize_to_corpus(text)
  vocab, merges = bpe_tokenize(corpus, 6, ["<|endoftext|>"])

  t = Tokenizer(vocab, merges, ["<|endoftext|>"])
  input = ['newest', 'lowest', 'low', 'widest']

  encoded = t.encode_iterable(input)
  expected = [262, 261, 260, 258, 260, 119, 105, 100, 258]
  for i, e in enumerate(encoded):
    assert e == expected[i]

  encoded = t.encode_iterable(input)
  assert ''.join(input) == b''.join([vocab[t] for t in encoded]).decode('utf-8')

def test_decode():
  text = b"low low low low low\nlower lower widest widest widest\nnewest newest newest newest newest newest"
  corpus = pretokenize_to_corpus(text)
  vocab, merges = bpe_tokenize(corpus, 6, ["<|endoftext|>"])

  t = Tokenizer(vocab, merges, ["<|endoftext|>"])
  assert t.decode([262, 261]) == "newest"

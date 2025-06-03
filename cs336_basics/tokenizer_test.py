from .tokenizer import bpe_tokenize

def test_basic():
  tokens = bpe_tokenize(b"low low low low low\nlower lower widest widest widest\nnewest newest newest newest newest newest", 6)
  assert len(tokens) == 263
  assert all(t in tokens for t in [b'st', b'est', b'ow', b'low', b'west', b'ne'])

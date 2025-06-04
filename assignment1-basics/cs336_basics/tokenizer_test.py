from .tokenizer import bpe_tokenize, pretokenize_to_corpus

def test_basic():
  text = b"low low low low low\nlower lower widest widest widest\nnewest newest newest newest newest newest"
  corpus = pretokenize_to_corpus(text)
  tokens, merges = bpe_tokenize(corpus, 6, ["<|endoftext|>"])
  assert len(tokens) == 263
  assert len(merges) == 6
  assert all(t in tokens.values() for t in [b'st', b'est', b'ow', b'low', b'west', b'ne'])

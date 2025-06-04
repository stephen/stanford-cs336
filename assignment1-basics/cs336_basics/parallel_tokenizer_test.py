from .parallel_pretokenizer import parallel_pretokenize_path_to_corpus
from .tokenizer import bpe_tokenize

def test_parallel_tokenize_e2e():
    # corpus = parallel_pretokenize_path_to_corpus("./data/TinyStoriesV2-GPT4-train.txt")
    corpus = parallel_pretokenize_path_to_corpus("./data/TinyStoriesV2-GPT4-valid.txt")

    tokens, merges = bpe_tokenize(corpus, 1000, ["<|endoftext|>"])
    assert len(merges) == 1000
    assert len(tokens) == 1000 + 1 + 256

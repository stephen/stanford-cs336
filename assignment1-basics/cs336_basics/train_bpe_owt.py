import pickle
import time
import pytest
from tqdm import tqdm
from .parallel_pretokenizer import parallel_pretokenize_path_to_corpus
from .tokenizer import bpe_tokenize

@pytest.mark.skip(reason="slow")
def test_tiny_stories_training_set():
    # corpus = parallel_pretokenize_path_to_corpus("./data/owt_valid.txt")
    corpus = parallel_pretokenize_path_to_corpus("./data/owt_train.txt")

    tokens, merges = bpe_tokenize(corpus, 10000)
    assert len(merges) == 10000
    assert len(tokens) == 10000 + 1 + 256
    with open("owt_train_vocab.pkl", "wb") as f:
        pickle.dump({
            "tokens": tokens,
            "merges": merges,
        }, f)

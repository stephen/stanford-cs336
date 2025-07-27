#!/bin/sh

# tensor parallel
uv run torchrun --nproc_per_node=2 ./cs336_basics/train_model.py --training_set=./data/TinyStoriesV2-GPT4-train.npy --validation_set=./data/TinyStoriesV2-GPT4-valid.npy --tokenizer_state=./data/TinyStoriesV2-GPT4-train.tokenizer_pkl --tp=2

# no parallel
# uv run torchrun ./cs336_basics/train_model.py --training_set=./data/TinyStoriesV2-GPT4-train.npy --validation_set=./data/TinyStoriesV2-GPT4-valid.npy --tokenizer_state=./data/TinyStoriesV2-GPT4-train.tokenizer_pkl --tp=1
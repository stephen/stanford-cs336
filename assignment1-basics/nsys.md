```bash
 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1 \
nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    -o "tp_profile_rank0" \
    # --capture-range=cudaProfilerApi \ # enable if t.cuda.profile() calls are in the training script
    uv run torchrun --nproc_per_node=2 \
        ./cs336_basics/train_model.py \
        --training_set=./data/TinyStoriesV2-GPT4-train.npy \
        --validation_set=./data/TinyStoriesV2-GPT4-valid.npy \
        --tokenizer_state=./data/TinyStoriesV2-GPT4-train.tokenizer_pkl \
        --tp=2 --steps=1
```
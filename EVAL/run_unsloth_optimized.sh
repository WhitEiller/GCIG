#!/bin/bash

# Kill any existing training process
pkill -f train_unsloth.py

# Clear cache to free memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Optimized Unsloth training with better performance settings
python training/train_unsloth.py \
    --model_name_or_path /mnt/disk/yh24/test1/bonito/Mistral-7B-v0.1 \
    --supervision_source bonito \
    --dataset_name pubmed_qa \
    --output_dir output/models/bonito_pubmed_qa_mistral_unsloth_optimized \
    --max_seq_length 2048 \
    --load_in_4bit \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --warmup_steps 10 \
    --max_steps 1000 \
    --logging_steps 10 \
    --save_steps 500 \
    --optim adamw_8bit \
    --weight_decay 0.01 \
    --lr_scheduler_type linear \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --use_gradient_checkpointing unsloth \
    --packing \
    --seed 3407 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --fp16 \
    --gradient_checkpointing_kwargs '{"use_reentrant": False}'
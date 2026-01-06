#!/bin/bash
# Test base model without fine-tuning
export CUDA_VISIBLE_DEVICES=0
python evaluation/evaluate_decoder.py \
    --dataset_name hotpot_qa \
    --model_name_or_path /mnt/disk/yh24/test1/Mistral-7B \
    --output_dir results/bio_qa_evaluation_base \
    --bf16
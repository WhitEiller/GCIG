#!/bin/bash
# Replace <checkpoint_path> with your actual checkpoint path
# Force use only GPU1 by using python directly instead of deepspeed
export CUDA_VISIBLE_DEVICES=0
python evaluation/evaluate_decoder.py \
    --dataset_name hotpot_qa \
    --model_name_or_path /mnt/disk/yh24/test1/Mistral-7B \
    --checkpoint_model_id_or_path /mnt/disk/yh24/test1/nayak-aclfindings24-code/output/bonito-hotpot/checkpoint-2335/adapter_model \
    --output_dir results/bio_qa_evaluation \
    --bf16

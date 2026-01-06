#!/bin/bash

# Fine-tune all 7 JSONL files using Unsloth training

BASE_DIR="/mnt/disk/yh24/test1/nayak-aclfindings24-code"
MODEL_PATH="/mnt/disk/yh24/test1/bonito/Mistral-7B-v0.1"
JSONL_DIR="${BASE_DIR}/jsonl"

# Array of JSONL files
JSONL_FILES=(
    "merged_sft1.jsonl"
    "merged_sft2.jsonl"
    "merged_sft3.jsonl"
    "question_outputs_mode1_sft.jsonl"
    "question_outputs_mode2_sft.jsonl"
    "question_outputs_mode3_sft.jsonl"
    "question_outputs_mode4_sft.jsonl"
)

# Training parameters
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
NUM_EPOCHS=1
LEARNING_RATE=2e-4
MAX_SEQ_LENGTH=2048

# Loop through each JSONL file and train
for jsonl_file in "${JSONL_FILES[@]}"; do
    # Extract filename without extension for output directory naming
    filename=$(basename "$jsonl_file" .jsonl)
    
    echo "=========================================="
    echo "Training on: ${jsonl_file}"
    echo "Output directory: ${BASE_DIR}/output/${filename}"
    echo "=========================================="
    
    python ${BASE_DIR}/training/train_unsloth.py \
        --model_name_or_path ${MODEL_PATH} \
        --custom_dataset_path ${JSONL_DIR}/${jsonl_file} \
        --output_dir ${BASE_DIR}/output/${filename} \
        --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --num_train_epochs ${NUM_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --load_in_4bit \
        --packing \
        --lora_r 16 \
        --lora_alpha 16 \
        --lora_dropout 0 \
        --warmup_steps 10 \
        --logging_steps 10 \
        --save_steps 1000 \
        --save_total_limit 1 \
        --seed 3407
    
    if [ $? -eq 0 ]; then
        echo "Successfully completed training for ${jsonl_file}"
    else
        echo "Error training ${jsonl_file}"
        exit 1
    fi
    
    echo ""
done

echo "=========================================="
echo "All training jobs completed!"
echo "=========================================="

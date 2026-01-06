#!/bin/bash

# 使用 deepspeed 和 train_decoder.py 对 jsonl 目录中的 5 个 JSONL 文件进行微调

BASE_DIR="/mnt/disk/yh24/test1/nayak-aclfindings24-code"
MODEL_PATH="/mnt/disk/yh24/test1/qwen2.5/Qwen2.5-7B-Instruct"
JSONL_DIR="${BASE_DIR}/jsonl"

# JSONL 文件数组
JSONL_FILES=(
    "question_outputs_mode1_sft.jsonl"
    "question_outputs_mode2_sft.jsonl"
    "question_outputs_mode3_sft.jsonl"
    "question_outputs_mode4_sft.jsonl"
    "synthetic_results_clean_sorted_top6000.jsonl"
)

# 训练参数
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16
NUM_EPOCHS=5
LEARNING_RATE=1e-4
BITS=4

# 循环处理每个 JSONL 文件
for jsonl_file in "${JSONL_FILES[@]}"; do
    # 提取文件名（不含扩展名）用于输出目录命名
    filename=$(basename "$jsonl_file" .jsonl)
    
    echo "=========================================="
    echo "开始训练: ${jsonl_file}"
    echo "输出目录: ${BASE_DIR}/output/qwen2.5_${filename}"
    echo "=========================================="
    
    deepspeed ${BASE_DIR}/training/train_decoder.py \
        --model_name_or_path ${MODEL_PATH} \
        --json_path ${JSONL_DIR}/${jsonl_file} \
        --output_dir ${BASE_DIR}/output/qwen2.5_${filename} \
        --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --num_train_epochs ${NUM_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --bf16 \
        --bits ${BITS} \
        --trust_remote_code
    
    # 检查训练是否成功
    if [ $? -eq 0 ]; then
        echo "✓ ${jsonl_file} 训练完成"
    else
        echo "✗ ${jsonl_file} 训练失败"
        exit 1
    fi
    
    echo ""
done

echo "=========================================="
echo "所有训练任务完成！"
echo "=========================================="

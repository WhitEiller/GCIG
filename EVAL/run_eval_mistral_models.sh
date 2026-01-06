#!/bin/bash

# 对微调的 Mistral 模型进行 HotpotQA 评测并将结果写入txt文件

BASE_DIR="/mnt/disk/yh24/test1/nayak-aclfindings24-code"
MODEL_PATH="/mnt/disk/yh24/test1/bonito/Mistral-7B-v0.1"
OUTPUT_BASE_DIR="${BASE_DIR}/output"
RESULTS_DIR="${BASE_DIR}/results/mistral_models_eval"
RESULTS_FILE="${RESULTS_DIR}/mistral_evaluation_results_summary.txt"

# 创建结果目录
mkdir -p ${RESULTS_DIR}

# 模型目录名称数组（对应 run_training_all_jsonl.sh 中训练的模型）
MODEL_DIRS=(
    "mistral_output_synthetic_dataset"
    "mistral_merged_sft1"
    "mistral_merged_sft2"
    "mistral_merged_sft3"
)

# 初始化结果文件
echo "======================================" > ${RESULTS_FILE}
echo "Mistral HotpotQA 评测结果汇总" >> ${RESULTS_FILE}
echo "评测时间: $(date)" >> ${RESULTS_FILE}
echo "======================================" >> ${RESULTS_FILE}
echo "" >> ${RESULTS_FILE}

# 循环处理每个模型
for model_dir in "${MODEL_DIRS[@]}"; do
    echo ""
    echo "=========================================="
    echo "开始评测模型: ${model_dir}"
    echo "=========================================="

    # 查找最新的checkpoint目录
    CHECKPOINT_DIR=$(find ${OUTPUT_BASE_DIR}/${model_dir} -name "checkpoint-*" -type d | sort -V | tail -n 1)

    if [ -z "${CHECKPOINT_DIR}" ]; then
        echo "✗ 未找到 ${model_dir} 的checkpoint目录，跳过"
        echo "模型: ${model_dir} - 状态: 未找到checkpoint" >> ${RESULTS_FILE}
        echo "--------------------" >> ${RESULTS_FILE}
        echo "" >> ${RESULTS_FILE}
        continue
    fi

    ADAPTER_PATH="${CHECKPOINT_DIR}/adapter_model"

    if [ ! -d "${ADAPTER_PATH}" ]; then
        echo "✗ 未找到 ${model_dir} 的adapter_model目录，跳过"
        echo "模型: ${model_dir} - 状态: 未找到adapter_model" >> ${RESULTS_FILE}
        echo "--------------------" >> ${RESULTS_FILE}
        echo "" >> ${RESULTS_FILE}
        continue
    fi

    echo "使用checkpoint: ${CHECKPOINT_DIR}"

    # 创建该模型的评测输出目录
    MODEL_RESULTS_DIR="${RESULTS_DIR}/${model_dir}"
    mkdir -p ${MODEL_RESULTS_DIR}

    # 执行评测
    export CUDA_VISIBLE_DEVICES=1
    python ${BASE_DIR}/evaluation/evaluate_decoder.py \
        --dataset_name hotpot_qa \
        --model_name_or_path ${MODEL_PATH} \
        --checkpoint_model_id_or_path ${ADAPTER_PATH} \
        --output_dir ${MODEL_RESULTS_DIR} \
        --bf16 \
        --trust_remote_code \
        2>&1 | tee ${MODEL_RESULTS_DIR}/eval_log.txt

    # 检查评测是否成功
    if [ $? -eq 0 ]; then
        echo "✓ ${model_dir} 评测完成"

        # 提取评测结果并追加到汇总文件
        echo "模型: ${model_dir}" >> ${RESULTS_FILE}
        echo "Checkpoint: ${CHECKPOINT_DIR}" >> ${RESULTS_FILE}

        # 查找并解析结果JSON文件
        RESULT_JSON=$(find ${MODEL_RESULTS_DIR} -name "results_*.json" | head -n 1)
        if [ -f "${RESULT_JSON}" ]; then
            echo "评测指标:" >> ${RESULTS_FILE}
            python3 -c "
import json
import sys
try:
    with open('${RESULT_JSON}', 'r') as f:
        data = json.load(f)
        eval_data = data.get('evaluation', {})
        print('  Accuracy: {:.4f}'.format(eval_data.get('accuracy', 0)))
        print('  F1: {:.4f}'.format(eval_data.get('f1', 0)))
        print('  ROUGE-F: {:.4f}'.format(eval_data.get('rouge_f', 0)))
except Exception as e:
    print('  无法解析结果文件:', str(e))
" >> ${RESULTS_FILE}
        else
            echo "  评测结果文件未生成" >> ${RESULTS_FILE}
        fi
    else
        echo "✗ ${model_dir} 评测失败"
        echo "模型: ${model_dir} - 状态: 评测失败" >> ${RESULTS_FILE}
    fi

    echo "--------------------" >> ${RESULTS_FILE}
    echo "" >> ${RESULTS_FILE}

    echo ""
done

echo "=========================================="
echo "所有评测任务完成！"
echo "结果汇总文件: ${RESULTS_FILE}"
echo "=========================================="

# 显示结果汇总
cat ${RESULTS_FILE}

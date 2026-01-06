#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
# Start vLLM OpenAI API server with LoRA adapter
vllm serve /mnt/disk/yh24/test1/bonito/Mistral-7B-v0.1 \
  --port 9991 \
  --served-model-name new \
  --host 0.0.0.0 \
  --enable-lora \
  --max-lora-rank 64 \
  --max-num-seqs 4 \
  --max-model-len 4096 \
  --disable-log-requests \
  --chat-template "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}<|input|>\n{{ message['content'].strip() }}\n<|output|>\n{% elif message['role'] == 'assistant' %}{{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}" \
  --lora-modules lora-mistral=/mnt/disk/yh24/test1/nayak-aclfindings24-code/output/new-hotpot/checkpoint-2825/adapter_model 
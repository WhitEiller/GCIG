#!/bin/bash

# Start model server with LoRA adapter
python model_server.py \
  --model_name_or_path /mnt/disk/yh24/test1/bonito/Mistral-7B-v0.1 \
  --checkpoint_model_id_or_path /mnt/disk/yh24/test1/nayak-aclfindings24-code/output/bonito-hotpot/checkpoint-2335/adapter_model \
  --bf16 \
  --port 9989 \
  --host 0.0.0.0
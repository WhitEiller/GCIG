#!/usr/bin/env python
"""Test script to check model dtype issues"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Model paths
model_path = "/mnt/disk/yh24/test1/bonito/Mistral-7B-v0.1"
checkpoint_path = "/mnt/disk/yh24/test1/nayak-aclfindings24-code/output/mistral-finetuned3/checkpoint-2685/adapter_model"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Loading PEFT adapter...")
model = PeftModel.from_pretrained(model, checkpoint_path)

print("\nModel dtype info:")
print(f"Model device: {model.device}")

# Check dtypes of various parts
for name, param in list(model.named_parameters())[:5]:
    print(f"{name}: {param.dtype}")

print("\nTesting generation...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

prompt = "Context: Test context.\n\nQuestion: What is a test?\n\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

print(f"Input dtype: {inputs['input_ids'].dtype}")

try:
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=False,
        )
    print("Generation successful!")
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated[:100]}...")
except Exception as e:
    print(f"Error during generation: {e}")
    print(f"Error type: {type(e)}")
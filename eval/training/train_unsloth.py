#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unsloth-optimized training script for Mistral model
Based on the original train_decoder.py but using Unsloth for better performance
"""

import os
import argparse
import torch
from datasets import load_dataset, DatasetDict
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from dataclasses import dataclass, field
from typing import Optional
import logging

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="mistralai/Mistral-7B-v0.1")
    max_seq_length: Optional[int] = field(default=2048)
    load_in_4bit: Optional[bool] = field(default=True)
    dtype: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    dataset_name: str = field(default="pubmed_qa")
    supervision_source: str = field(default="bonito")
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    custom_dataset_path: Optional[str] = field(default=None)

@dataclass
class UnslothTrainingArguments:
    output_dir: str = field(default="./output_unsloth")
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-4)
    warmup_steps: int = field(default=10)
    max_steps: int = field(default=10000)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=10000)
    save_total_limit: int = field(default=1)
    fp16: bool = field(default=not torch.cuda.is_bf16_supported())
    bf16: bool = field(default=torch.cuda.is_bf16_supported())
    optim: str = field(default="adamw_8bit")
    weight_decay: float = field(default=0.01)
    lr_scheduler_type: str = field(default="linear")
    seed: int = field(default=3407)
    
    # LoRA parameters
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0)
    
    # Unsloth specific
    use_gradient_checkpointing: str = field(default="unsloth")
    packing: bool = field(default=True)

def load_and_prepare_model(args):
    """Load model using Unsloth's FastLanguageModel"""
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )
    
    # Add LoRA adapters using Unsloth
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
    )
    
    return model, tokenizer

def load_training_dataset(data_args):
    """Load dataset from HuggingFace or JSONL file"""
    
    if data_args.custom_dataset_path:
        # Check if it's a JSONL file
        if data_args.custom_dataset_path.endswith('.jsonl') or data_args.custom_dataset_path.endswith('.json'):
            dataset = load_dataset("json", data_files=data_args.custom_dataset_path)
            # Rename columns if needed (question -> input, answer -> output)
            if "train" in dataset and "question" in dataset["train"].column_names:
                dataset = dataset.rename_columns({"question": "input", "answer": "output"})
            # Create a train split if it doesn't exist
            if "train" not in dataset:
                dataset = DatasetDict({"train": dataset["train" if "train" in dataset else list(dataset.keys())[0]]})
        else:
            dataset = load_dataset(data_args.custom_dataset_path)
    else:
        # Load from BatsResearch/bonito-experiment
        config_name = f"{data_args.supervision_source}_{data_args.dataset_name}"
        dataset = load_dataset("BatsResearch/bonito-experiment", config_name)
    
    return dataset

def format_dataset_for_unsloth(dataset, tokenizer, model_type="mistral"):
    """Format dataset for Unsloth training"""
    
    def format_prompts(examples):
        texts = []
        for i in range(len(examples["input"])):
            if model_type == "mistral":
                # Mistral format
                text = f"{tokenizer.bos_token}<|input|>\n{examples['input'][i].strip()}\n<|output|>\n{examples['output'][i].strip()}{tokenizer.eos_token}"
            else:
                # Default format
                text = f"{tokenizer.bos_token}<|input|>\n{examples['input'][i].strip()}\n<|output|>\n{examples['output'][i].strip()}{tokenizer.eos_token}"
            texts.append(text)
        return {"text": texts}
    
    # Apply formatting
    if "train" in dataset:
        dataset["train"] = dataset["train"].map(
            format_prompts,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
    
    if "validation" in dataset or "eval" in dataset:
        eval_key = "validation" if "validation" in dataset else "eval"
        dataset[eval_key] = dataset[eval_key].map(
            format_prompts,
            batched=True,
            remove_columns=dataset[eval_key].column_names
        )
    
    return dataset

def main():
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--dtype", type=str, default=None)
    
    # Data arguments
    parser.add_argument("--dataset_name", type=str, default="pubmed_qa")
    parser.add_argument("--supervision_source", type=str, default="bonito")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--custom_dataset_path", type=str, default=None)
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./output_unsloth")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", default=not torch.cuda.is_bf16_supported())
    parser.add_argument("--bf16", action="store_true", default=torch.cuda.is_bf16_supported())
    parser.add_argument("--optim", type=str, default="adamw_8bit")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--seed", type=int, default=3407)
    
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0)
    
    # Unsloth specific
    parser.add_argument("--use_gradient_checkpointing", type=str, default="unsloth")
    parser.add_argument("--packing", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    logger.info("Loading model with Unsloth...")
    model, tokenizer = load_and_prepare_model(args)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    logger.info(f"Total params: {total_params:,}")
    
    logger.info("Loading dataset...")
    dataset = load_training_dataset(args)
    
    logger.info("Formatting dataset for Unsloth...")
    dataset = format_dataset_for_unsloth(dataset, tokenizer, model_type="mistral")
    
    # Prepare train and eval datasets
    train_dataset = dataset["train"]
    eval_dataset = None
    
    if "validation" in dataset:
        eval_dataset = dataset["validation"]
    elif "eval" in dataset:
        eval_dataset = dataset["eval"]
    else:
        # Split train dataset if no eval dataset exists
        split_dataset = train_dataset.train_test_split(test_size=0.01, seed=args.seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    
    # Apply sample limits if specified
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    
    if args.max_eval_samples and eval_dataset:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=False,
        group_by_length=True,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=args.packing,
        args=training_args,
    )
    
    logger.info("Starting training with Unsloth...")
    logger.info(f"Number of training samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Number of evaluation samples: {len(eval_dataset)}")
    
    # Train the model
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {args.output_dir}")
    model.save_pretrained_merged(args.output_dir, tokenizer, save_method="lora")
    
    # Also save just the LoRA adapters
    lora_output_dir = os.path.join(args.output_dir, "lora_adapters")
    model.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)
    
    logger.info("Training completed successfully!")
    
    # Optional: Push to hub if needed
    # model.push_to_hub_merged("username/model-name", tokenizer, save_method="lora")

if __name__ == "__main__":
    main()
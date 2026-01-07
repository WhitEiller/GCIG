#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
import torch
import time
import uuid
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    LlamaTokenizer,
)
from peft import PeftModel
from peft.tuners.lora import LoraLayer
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="mistralai/Mistral-7B-v0.1",
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    checkpoint_model_id_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The LoRA checkpoint path."}
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Use bfloat16 precision."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache directory for models."}
    )

@dataclass
class TrainingArguments:
    output_dir: str = field(default="temp")
    bf16: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)

def get_model(args):
    device_map = None
    if torch.cuda.is_available():
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            device_map = {"": local_rank}
        else:
            device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        use_auth_token=args.use_auth_token if hasattr(args, 'use_auth_token') else False,
        trust_remote_code=args.trust_remote_code,
        device_map=device_map,
    )
    return model

def get_tokenizer(model, args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=True if "pythia" in args.model_name_or_path else False,
        tokenizer_type="llama" if "llama" in args.model_name_or_path else None,
        use_auth_token=args.use_auth_token if hasattr(args, 'use_auth_token') else False,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if "llama" in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        logger.info("Adding special tokens.")
        model.config.pad_token_id = 0
        tokenizer.add_special_tokens({
            "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
            "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
            "unk_token": tokenizer.convert_ids_to_tokens(
                model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
            ),
        })
    return tokenizer

def get_peft_model(model, args):
    model = PeftModel.from_pretrained(
        model, args.checkpoint_model_id_or_path, is_trainable=False
    )

    target_dtype = torch.bfloat16 if args.bf16 else torch.float32

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(target_dtype)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(target_dtype)

    if hasattr(model, 'base_model'):
        model.base_model = model.base_model.to(target_dtype)

    return model

class ModelServer:
    def __init__(self, model_args):
        logger.info("Initializing model server...")

        # Load model using existing functions
        self.model = get_model(model_args)

        # Load LoRA adapter if specified
        if model_args.checkpoint_model_id_or_path:
            logger.info(f"Loading LoRA adapter from {model_args.checkpoint_model_id_or_path}")
            self.model = get_peft_model(self.model, model_args)

        # Get tokenizer
        self.tokenizer = get_tokenizer(self.model, model_args)

        # Set model to evaluation mode
        self.model.eval()

        # Store model args for prompt formatting
        self.model_args = model_args

        logger.info("Model server initialized successfully!")

    def format_prompt(self, input_text, simple_prompt=False):
        """Format prompt based on model type"""
        if simple_prompt or (
            self.model_args.checkpoint_model_id_or_path is None and
            (self.model_args.model_name_or_path == "mistralai/Mistral-7B-v0.1" or
             self.model_args.model_name_or_path == "meta-llama/Llama-2-7b-hf")
        ):
            # Simple prompt format for base models
            return f"{self.tokenizer.bos_token}Instruct: {input_text.strip()}\nAnswer:"
        elif self.model_args.model_name_or_path == "mistralai/Mistral-7B-Instruct-v0.2":
            # Mistral instruct format
            return f"{self.tokenizer.bos_token}[INST] {input_text.strip()} [/INST]"
        else:
            # Default format for fine-tuned models
            return f"{self.tokenizer.bos_token}<|input|>\n{input_text.strip()}\n<|output|>\n"

    def generate(self, prompt, max_new_tokens=512, temperature=0.1, do_sample=False, simple_prompt=False):
        """Generate response using the model"""
        try:
            # Format the prompt
            formatted_prompt = self.format_prompt(prompt, simple_prompt)

            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )

            # Move inputs to model device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the generated part
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise e

# Global model server instance
model_server = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Model server is running"})

@app.route('/generate', methods=['POST'])
def generate_text():
    """Generate text endpoint"""
    try:
        data = request.get_json()

        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400

        prompt = data['prompt']
        max_new_tokens = data.get('max_new_tokens', 512)
        temperature = data.get('temperature', 0.1)
        do_sample = data.get('do_sample', False)
        simple_prompt = data.get('simple_prompt', False)

        # Generate response
        response = model_server.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            simple_prompt=simple_prompt
        )

        return jsonify({
            "prompt": prompt,
            "response": response,
            "metadata": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "simple_prompt": simple_prompt
            }
        })

    except Exception as e:
        logger.error(f"Error in generate_text: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        "model_name": "bonito",
        "base_model": model_server.model_args.model_name_or_path,
        "checkpoint_path": model_server.model_args.checkpoint_model_id_or_path,
        "device": str(next(model_server.model.parameters()).device),
        "dtype": str(next(model_server.model.parameters()).dtype)
    })

# OpenAI API compatible endpoints
@app.route('/v1/models', methods=['GET'])
def list_models():
    """OpenAI compatible models endpoint"""
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "bonito",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "permission": [],
                "root": "bonito",
                "parent": None
            }
        ]
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI compatible chat completions endpoint"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": {"message": "Invalid JSON in request body", "type": "invalid_request_error"}}), 400

        # Extract parameters
        messages = data.get('messages', [])
        model = data.get('model', 'bonito')
        max_tokens = data.get('max_tokens', 512)
        temperature = data.get('temperature', 0.1)
        stream = data.get('stream', False)

        if not messages:
            return jsonify({"error": {"message": "Missing required parameter: messages", "type": "invalid_request_error"}}), 400

        # Convert messages to a single prompt
        prompt = ""
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')

            if role == 'system':
                prompt += f"System: {content}\n"
            elif role == 'user':
                prompt += f"User: {content}\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n"

        # Add final prompt for assistant response
        prompt += "Assistant:"

        # Generate response
        try:
            response_text = model_server.generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                simple_prompt=False
            )
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return jsonify({"error": {"message": f"Generation failed: {str(e)}", "type": "internal_error"}}), 500

        # Create OpenAI compatible response
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split())
            }
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in chat_completions: {str(e)}")
        return jsonify({"error": {"message": str(e), "type": "internal_error"}}), 500

@app.route('/v1/completions', methods=['POST'])
def completions():
    """OpenAI compatible completions endpoint"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": {"message": "Invalid JSON in request body", "type": "invalid_request_error"}}), 400

        # Extract parameters
        prompt = data.get('prompt', '')
        model = data.get('model', 'bonito')
        max_tokens = data.get('max_tokens', 512)
        temperature = data.get('temperature', 0.1)

        if not prompt:
            return jsonify({"error": {"message": "Missing required parameter: prompt", "type": "invalid_request_error"}}), 400

        # Generate response
        try:
            response_text = model_server.generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                simple_prompt=False
            )
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return jsonify({"error": {"message": f"Generation failed: {str(e)}", "type": "internal_error"}}), 500

        # Create OpenAI compatible response
        response = {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": response_text,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split())
            }
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in completions: {str(e)}")
        return jsonify({"error": {"message": str(e), "type": "internal_error"}}), 500

def main():
    parser = argparse.ArgumentParser(description="Model Server")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--model_name_or_path", type=str, default="mistralai/Mistral-7B-v0.1",
                       help="Model name or path")
    parser.add_argument("--checkpoint_model_id_or_path", type=str, default=None,
                       help="LoRA checkpoint path")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--port", type=int, default=9989, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")

    args = parser.parse_args()

    # Parse config file if provided
    if args.config and os.path.exists(args.config):
        hf_parser = HfArgumentParser((ModelArguments, TrainingArguments))
        model_args, training_args = hf_parser.parse_json_file(json_file=args.config)
    else:
        # Create args from command line
        model_args = ModelArguments(
            model_name_or_path=args.model_name_or_path,
            checkpoint_model_id_or_path=args.checkpoint_model_id_or_path,
            bf16=args.bf16
        )
        training_args = TrainingArguments(
            output_dir="temp",
            bf16=args.bf16
        )

    # Initialize global model server
    global model_server
    model_server = ModelServer(model_args)

    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)

if __name__ == "__main__":
    main()
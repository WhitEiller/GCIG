import argparse
import json
import os
import random
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import datasets
import evaluate
from rouge_score import rouge_scorer
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from peft.tuners.lora import LoraLayer
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(2, str(Path(__file__).resolve().parents[1]))

import copy
from typing import Dict, Optional, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from templates import choose_template, gather_templates

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DEFAULT_PAD_TOKEN = "[PAD]"
IGNORE_INDEX = -100


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )

    bits: int = field(default=4, metadata={"help": "How many bits to use."})

    max_memory_MB: int = field(default=80000, metadata={"help": "Free memory per gpu."})

    output_dir: str = field(
        default="results", metadata={"help": "The output dir for logs and checkpoints"}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Removed unused columns. Needed to make this codebase work."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        default="mistralai/Mistral-7B-v0.1",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    checkpoint_model_id_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."},
    )


@dataclass
class TestArguments:
    dataset_name: str = field(
        default="pubmed_qa",
        metadata={
            "help": "The name of the dataset to use (via the datasets library).",
        },
    )
    template_name: str = field(
        default=None,
        metadata={
            "help": "The template/prompt name",
        },
    )
    simple_prompt: bool = field(
        default=False,
        metadata={
            "help": "Use simple prompt",
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={
            "help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    target_max_len: int = field(
        default=1024,
        metadata={
            "help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    test_debug: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Activate debug mode and run training only with a subset of data.",
        },
    )
    api_url: Optional[str] = field(
        default=None,
        metadata={
            "help": "API URL for remote model evaluation (e.g., OpenAI-compatible API endpoint).",
        },
    )
    model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Model name to use when calling the API endpoint.",
        },
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


def get_model(args):
    # Force GPU1 usage
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    device_map = None
    max_memory = None

    # For deepspeed, let it handle device placement
    if torch.cuda.is_available():
        if torch.distributed.is_initialized():
            # In distributed mode, use GPU1 (which becomes GPU0 after CUDA_VISIBLE_DEVICES)
            local_rank = 0
            device_map = {"": local_rank}
        else:
            device_map = {"": 0}  # Use GPU0 (which is actually GPU1 due to CUDA_VISIBLE_DEVICES)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        use_auth_token=True,
        trust_remote_code=args.trust_remote_code,
        device_map=device_map,
    )

    return model


def get_tokenizer(model, args):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=True if "pythia" in args.model_name_or_path else False,  # Fast tokenizer giving issues.
        tokenizer_type="llama"
        if "llama" in args.model_name_or_path
        else None,  # Needed for HF name change
        use_auth_token=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    if "llama" in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        logger.info("Adding special tokens.")
        model.config.pad_token_id = 0  # avoid getting NoneType error
        tokenizer.add_special_tokens(
            {
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id
                    if model.config.pad_token_id != -1
                    else tokenizer.pad_token_id
                ),
            }
        )

    return tokenizer


def get_peft_model(model, args):
    model = PeftModel.from_pretrained(
        model, args.checkpoint_model_id_or_path, is_trainable=False
    )
    
    # Ensure all model components use the correct dtype
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
    
    # Ensure the base model is also in the correct dtype
    if hasattr(model, 'base_model'):
        model.base_model = model.base_model.to(target_dtype)
    
    return model


def call_api(prompt: str, api_url: str, model_name: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
    """Call OpenAI-compatible API endpoint for text generation."""
    try:
        # Try the endpoint - vLLM uses /v1/completions endpoint
        endpoint = f"{api_url}/completions"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["text"].strip()
    except requests.exceptions.HTTPError as e:
        logger.error(f"API call failed with status {response.status_code}: {e}")
        logger.error(f"Response content: {response.text}")
        logger.error(f"Request payload: {payload}")
        return ""
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return ""


@dataclass
class DataCollatorForEvaluation(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    model_name_or_path: str
    simple_prompt: bool = False

    def add_input_prompt(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if self.simple_prompt:
            # obtained from phi; this will be used for base models and base model + tapt
            sources = [
                f"{self.tokenizer.bos_token}Instruct: {example['input'].strip()}\nAnswer:"
                for example in instances
                for _ in range(len(example["answer_choices"]))
            ]
        elif self.model_name_or_path == "mistralai/Mistral-7B-Instruct-v0.2":
            sources = [
                f"{self.tokenizer.bos_token}[INST] {example['input'].strip()} [/INST]"
                for example in instances
                for _ in range(len(example["answer_choices"]))
            ]
        else:
            # for the rest we explicitly train the model with this prompt format.
            sources = [
                f"{self.tokenizer.bos_token}<|input|>\n{example['input'].strip()}\n<|output|>\n"
                for example in instances
                for _ in range(len(example["answer_choices"]))
            ]

        return sources

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # flatten with answer choices
        sources = self.add_input_prompt(instances)

        if self.model_name_or_path == "mistralai/Mistral-7B-Instruct-v0.2":
            labels = [
                f"{ans.strip()}"
                for example in instances
                for ans in example["answer_choices"]
            ]
        else:
            labels = [
                f"{ans.strip()}{self.tokenizer.eos_token}"
                for example in instances
                for ans in example["answer_choices"]
            ]

        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_labels = self.tokenizer(
            labels,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        label_ids = []
        for tokenized_source, tokenized_label in zip(
            tokenized_sources_with_prompt["input_ids"],
            tokenized_labels["input_ids"],
        ):
            input_ids.append(torch.tensor(tokenized_source + tokenized_label))
            label_ids.append(
                torch.tensor(
                    [IGNORE_INDEX for _ in range(len(tokenized_source))]
                    + copy.deepcopy(tokenized_label)
                )
            )
        # Apply padding
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        label_ids = pad_sequence(
            label_ids, batch_first=True, padding_value=IGNORE_INDEX
        )
        data_dict = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "labels": label_ids,
            "targets": torch.tensor([ex["label"] for ex in instances]),
        }
        return data_dict


class RankedInferenceTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        model_inputs = {k: inputs[k] for k in ["input_ids", "attention_mask", "labels"]}
        _logits = model(**model_inputs).logits
        targets = inputs.pop("targets")
        labels = inputs.pop("labels")

        # causal language model loss (this is from huggingface)
        _logits = _logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        log_probs = nn.functional.log_softmax(_logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(IGNORE_INDEX)
        labels = torch.clamp(labels, min=0)
        log_probs = log_probs.gather(dim=-1, index=labels)

        log_probs.masked_fill_(padding_mask, 0.0)

        seq_log_probs = log_probs.squeeze(-1).sum(-1)
        output_logits = seq_log_probs.view(targets.size(0), -1)

        loss = nn.CrossEntropyLoss()(output_logits, targets)

        return (loss, {"logits": output_logits}) if return_outputs else loss


def evaluate_template(
    model, tokenizer, dataset, template, data_collator, training_args, api_url=None, model_name=None
):
    column_names = dataset.column_names
    # Check if this is a generation task
    is_generation = template.get_answer_choices_list({k: dataset[0][k] for k in column_names}) is None

    def preprocess_function(examples):
        bs = len(examples[column_names[0]])

        input_texts = []
        target_texts = []
        answer_choices_texts = []

        for i in range(bs):
            ex = {k: examples[k][i] for k in column_names}
            input, target = template.apply(ex)
            input_texts.append(input)
            target_texts.append(target)

            # For generation tasks, answer_choices might be None
            answer_choices = template.get_answer_choices_list(ex)
            if answer_choices:
                answer_choices_texts.append(answer_choices)
            else:
                # For generation task, use the target as the only choice
                answer_choices_texts.append([target])

        # For generation tasks, label is always 0 (we'll evaluate differently)
        if answer_choices_texts and len(answer_choices_texts[0]) > 1:
            targets = [
                answer_choices_texts[idx].index(t) for idx, t in enumerate(target_texts)
            ]
        else:
            targets = [0] * len(target_texts)  # Default to 0 for generation

        return {
            "input": input_texts,
            "answer_choices": answer_choices_texts,
            "label": targets,
            "ground_truth": target_texts,  # Store ground truth for partial matching
        }

    eval_dataset = dataset.map(
        preprocess_function, batched=True, remove_columns=column_names
    )

    logger.info(eval_dataset[0])

    if is_generation or api_url:
        # For generation tasks or API-based inference, generate text and evaluate with partial matching
        generated_texts = []
        ground_truths = eval_dataset["ground_truth"]

        # Prepare all prompts first
        all_prompts = []
        for i in range(len(eval_dataset)):
            ex = eval_dataset[i]
            input_text = ex["input"]

            # Add proper prompt format
            if data_collator.simple_prompt:
                prompt = f"Instruct: {input_text.strip()}\nAnswer:"
            elif data_collator.model_name_or_path == "mistralai/Mistral-7B-Instruct-v0.2":
                prompt = f"[INST] {input_text.strip()} [/INST]"
            else:
                prompt = f"<|input|>\n{input_text.strip()}\n<|output|>\n"

            if not api_url and tokenizer.bos_token:
                prompt = f"{tokenizer.bos_token}{prompt}"

            all_prompts.append(prompt)

        if api_url:
            # Use API for inference (still sequential for API)
            for prompt in tqdm(all_prompts, desc="Generating responses (API)"):
                generated = call_api(
                    prompt=prompt,
                    api_url=api_url,
                    model_name=model_name,
                    max_tokens=data_collator.target_max_len,
                    temperature=0.1
                )
                generated_texts.append(generated)
        else:
            # Use local model for batched inference
            batch_size = 8  # Adjust based on GPU memory

            # Save original padding side and switch to left for generation
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "left"

            for batch_start in tqdm(range(0, len(all_prompts), batch_size), desc="Generating responses"):
                batch_end = min(batch_start + batch_size, len(all_prompts))
                batch_prompts = all_prompts[batch_start:batch_end]

                # Tokenize batch with padding
                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=data_collator.source_max_len,
                    padding=True
                )

                # Move inputs to the correct device
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                input_length = inputs["input_ids"].shape[1]  # Total input length including padding

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,  # Reduced for speed, adjust as needed
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                # Decode only the generated part for each sample
                for output in outputs:
                    generated = tokenizer.decode(output[input_length:], skip_special_tokens=True)
                    generated_texts.append(generated)

            # Restore original padding side
            tokenizer.padding_side = original_padding_side

        return None, None, generated_texts, ground_truths
    else:
        # Original logic for classification
        training_args.label_names = ["targets"]
        trainer = RankedInferenceTrainer(
            model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        logits, targets, _ = trainer.predict(eval_dataset)
        return logits, targets, None, None


def compute_metrics(logits, targets, generated_texts=None, ground_truths=None):
    # For generation tasks with partial matching
    if generated_texts is not None and ground_truths is not None:
        # Partial matching: check if ground truth is contained in generated text
        correct = 0
        for gen_text, ground_truth in zip(generated_texts, ground_truths):
            # Case-insensitive partial matching
            if ground_truth.lower().strip() in gen_text.lower():
                correct += 1

        accuracy = correct / len(generated_texts)
        # For binary classification based on partial match
        predictions = [1 if ground_truth.lower().strip() in gen_text.lower() else 0
                      for gen_text, ground_truth in zip(generated_texts, ground_truths)]
        references = [1] * len(generated_texts)  # All should match

        metrics = {}
        metrics['accuracy'] = accuracy

        # Calculate F1 for the partial matching
        f1_metric = evaluate.load("f1")
        metrics.update(
            f1_metric.compute(predictions=predictions, references=references, average="macro")
        )

        # Calculate ROUGE-F scores
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_f_scores = []
        for gen_text, ground_truth in zip(generated_texts, ground_truths):
            score = scorer.score(ground_truth, gen_text)
            rouge_f_scores.append(score['rougeL'].fmeasure)

        metrics['rouge_f'] = sum(rouge_f_scores) / len(rouge_f_scores)
    else:
        # Original logic for classification tasks
        predictions = np.argmax(logits, axis=-1)

        metrics = {}
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")

        metrics.update(accuracy_metric.compute(predictions=predictions, references=targets))
        metrics.update(
            f1_metric.compute(predictions=predictions, references=targets, average="macro")
        )

        # For classification tasks, ROUGE-F is not applicable, set to 0
        metrics['rouge_f'] = 0.0

    return metrics


def save_results(metrics, dataset_name, template, targets, predictions, output_dir, generated_texts=None, ground_truths=None):
    template_name = template.name
    _template_name = re.sub("\s+", "_", template_name.lower())

    if generated_texts is not None and ground_truths is not None:
        # Save generation results
        pd.DataFrame({"ground_truth": ground_truths, "generated": generated_texts}).to_csv(
            os.path.join(
                output_dir,
                "template_" + _template_name + "_generation_results.csv",
            ),
            index=False,
        )
    else:
        pd.DataFrame(targets).to_csv(
            os.path.join(
                output_dir,
                "template_" + _template_name + "_target.csv",
            ),
            index=False,
            header=False,
        )

        pd.DataFrame(predictions).to_csv(
            os.path.join(
                output_dir,
                "template_" + _template_name + "_predict.csv",
            ),
            index=False,
            header=False,
        )

    results = {
        "dataset_name": dataset_name,
        "hf_dataset_name": f"BatsResearch/bonito-experiment-eval/{dataset_name}",
        "template_name": template_name,
        "evaluation": {
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "rouge_f": metrics["rouge_f"],
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, f"results_{_template_name}.json"),
        "w",
    ) as f:
        json.dump(results, f, indent=4)


def main():
    parser = HfArgumentParser((ModelArguments, TestArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, test_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, test_args, training_args = parser.parse_args_into_dataclasses()

    args = argparse.Namespace(
        **vars(model_args), **vars(test_args), **vars(training_args)
    )
    # text dataset
    if args.dataset_name == "bio_qa":
        # Load local JSON dataset
        import json
        with open("data/bio_qa_dataset_clean.json", "r") as f:
            data = json.load(f)

        # Convert to HuggingFace dataset format
        from datasets import Dataset
        text_dataset = Dataset.from_list(data)
    elif args.dataset_name == "hotpot_qa":
        # Load HotpotQA dataset
        import json
        with open("//test1/graphrag-purity/HotpotEval_Test.json", "r") as f:
            data = json.load(f)

        # Convert to the expected format for evaluation
        processed_data = []
        for item in data:
            processed_data.append({
                "question": item["question"],
                "answer": item["answer"]
            })
        processed_data = processed_data[:1000]
        from datasets import Dataset
        text_dataset = Dataset.from_list(processed_data)
    elif args.dataset_name == "privacy_qa":
        # Load local Arrow dataset
        from datasets import Dataset
        import pyarrow as pa

        # Read the arrow file
        with pa.memory_map("privacy_qa.arrow", "r") as source:
            text_dataset = Dataset(pa.ipc.open_stream(source).read_all())

        # Limit to first 1000 samples for testing
        text_dataset = text_dataset.select(range(min(len(text_dataset), 1000)))
    else:
        text_dataset = load_dataset(
            "BatsResearch/bonito-experiment-eval", args.dataset_name
        )["test"]

    # Trim a number of evaluation examples
    if args.test_debug:
        text_dataset = text_dataset.select(range(min(len(text_dataset), 10)))

    column_names = text_dataset.column_names

    # Skip model/tokenizer loading if using API
    if args.api_url:
        logger.info(f"Using API endpoint: {args.api_url} with model: {args.model_name}")
        model = None
        tokenizer = None
        # Create a minimal data collator for API mode
        class SimpleDataCollator:
            def __init__(self):
                self.source_max_len = args.source_max_len
                self.target_max_len = args.target_max_len
                self.model_name_or_path = args.model_name if args.model_name else ""
                self.simple_prompt = args.simple_prompt

        data_collator = SimpleDataCollator()
    else:
        model = get_model(args)
        if args.checkpoint_model_id_or_path:
            model = get_peft_model(model, args)

        # Ensure model is in the correct dtype for generation
        if args.bf16:
            model = model.to(torch.bfloat16)

        model.eval()
        tokenizer = get_tokenizer(model, args)
        if args.checkpoint_model_id_or_path is None and (
            args.model_name_or_path == "mistralai/Mistral-7B-v0.1"
            or args.model_name_or_path == "meta-llama/Llama-2-7b-hf"
        ):
            args.simple_prompt = True

        data_collator = DataCollatorForEvaluation(
            tokenizer=tokenizer,
            source_max_len=args.source_max_len,
            target_max_len=args.target_max_len,
            model_name_or_path=args.model_name_or_path,
            simple_prompt=args.simple_prompt,
        )

    if args.template_name:
        if args.dataset_name == "hotpot_qa":
            # Use custom HotpotQA template
            from templates.templates.hotpot_qa import HotpotQATemplate
            templates = [HotpotQATemplate()]
        elif args.dataset_name == "privacy_qa":
            # Use custom Privacy QA template
            from templates.templates.privacy_qa_template import PrivacyQATemplate
            templates = [PrivacyQATemplate()]
        else:
            templates = [
                choose_template(
                    "BatsResearch/bonito-experiment-eval",
                    args.dataset_name,
                    args.template_name,
                )
            ]
    else:
        if args.dataset_name == "hotpot_qa":
            # Use custom HotpotQA template
            from templates.templates.hotpot_qa import HotpotQATemplate
            templates = [HotpotQATemplate()]
        elif args.dataset_name == "privacy_qa":
            # Use custom Privacy QA template
            from templates.templates.privacy_qa_template import PrivacyQATemplate
            templates = [PrivacyQATemplate()]
        else:
            logger.info(
                f"No template_name given; Evaluating all templates for {args.dataset_name}"
            )

            templates = []
            for template_name in gather_templates()[
                ("BatsResearch/bonito-experiment-eval", args.dataset_name)
            ].keys():
                templates.append(
                    choose_template(
                        "BatsResearch/bonito-experiment-eval",
                        args.dataset_name,
                        template_name,
                    )
                )

    for template in templates:
        with training_args.main_process_first("main local process"):
            logger.info(f"Dataset {args.dataset_name} Template {template.name}")

        logits, targets, generated_texts, ground_truths = evaluate_template(
            model, tokenizer, text_dataset, template, data_collator, training_args,
            api_url=args.api_url, model_name=args.model_name
        )

        metrics = compute_metrics(logits, targets, generated_texts, ground_truths)
        logger.info(metrics)

        if training_args.should_save and training_args.local_rank in [-1, 0]:
            save_results(
                metrics,
                args.dataset_name,
                template,
                targets,
                logits,
                training_args.output_dir,
                generated_texts,
                ground_truths,
            )
    logger.info("Done!")


if __name__ == "__main__":
    main()

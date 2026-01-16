#!/usr/bin/env python3
"""
SFT trainer for personality trait-enhanced fine-tuning.

Supports both full fine-tuning and QLoRA (configurable via YAML).

- Reads a YAML config specifying model paths, data, and training hyperparameters
- Loads a locally downloaded base model (e.g., Qwen/Qwen3-14B snapshot)
- Prepares dataset from JSON/JSONL with either `messages` or `dialogue`
- Applies a generic chat template (fallback) and optional system message
- Runs TRL SFT with optional LoRA (4-bit quantization) or full fine-tuning
- Saves model weights and training configs

Usage:
  python train_sft.py --config train.dgx.yaml
"""

import os
import json
import yaml
import math
import argparse
import random
import inspect
from typing import List, Dict, Any

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import EarlyStoppingCallback


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_messages_from_dialogue(dialogue: List[str]) -> List[Dict[str, str]]:
    messages = []
    for i, utt in enumerate(dialogue):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": utt})
    return messages


def maybe_insert_system_message(messages: List[Dict[str, str]], system_text: str) -> List[Dict[str, str]]:
    if not messages:
        return messages
    if messages[0]["role"] != "system":
        return [{"role": "system", "content": system_text}] + messages
    return messages


def json_to_hf_dataset(
    json_path: str,
    val_size,  # Union[int, float] - absolute count or percentage (0.0-1.0)
    add_system: bool,
    system_message: str,
    seed: int,
) -> DatasetDict:
    # Load data - support both JSON array and JSONL (one JSON per line)
    data = []
    with open(json_path, "r") as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError(f"Empty file: {json_path}")
        
        # Try to detect format
        f.seek(0)
        try:
            # Try loading as single JSON (array or object)
            content = f.read()
            parsed = json.loads(content)
            if isinstance(parsed, list):
                data = parsed
            else:
                data = [parsed]
        except json.JSONDecodeError:
            # Likely JSONL format - one JSON object per line
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed line: {e}")
                        continue

    records: List[Dict[str, Any]] = []
    for entry in data:
        if "messages" in entry and isinstance(entry["messages"], list):
            messages = entry["messages"]
        elif "dialogue" in entry and isinstance(entry["dialogue"], list):
            messages = build_messages_from_dialogue(entry["dialogue"])
        else:
            # Skip malformed entries
            continue

        if add_system:
            messages = maybe_insert_system_message(messages, system_message)

        records.append({"messages": messages})

    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(records)

    # Validation split - support both absolute count and percentage
    n = len(records)
    if n == 0:
        raise ValueError("No valid records found in input JSON.")
    
    # Convert percentage to absolute count if needed
    if isinstance(val_size, float) and 0.0 < val_size < 1.0:
        # Percentage mode (e.g., 0.1 = 10%)
        val_count = int(n * val_size)
        print(f"Using {val_size*100:.1f}% of data for validation: {val_count} samples")
    elif isinstance(val_size, (int, float)) and val_size >= 1:
        # Absolute count mode
        val_count = int(val_size)
        # Cap at 20% max for safety
        val_count = min(val_count, n // 5)
        print(f"Using {val_count} samples for validation ({val_count/n*100:.1f}% of data)")
    else:
        val_count = 0
    
    val_count = max(0, min(val_count, n - 1)) if val_count > 0 else 0

    eval_data = records[:val_count] if val_count > 0 else []
    train_data = records[val_count:] if val_count > 0 else records

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data) if eval_data else Dataset.from_list([])

    return DatasetDict({"train": train_dataset, "test": eval_dataset})


def apply_template_and_tokenize(dataset: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
    # Fallback generic chat template if tokenizer lacks one
    if not getattr(tokenizer, "chat_template", None):
        DEFAULT_CHAT_TEMPLATE = (
            "{% for message in messages %}\n"
            "{% if message['role'] == 'user' %}\n{{ '\n' + message['content'] + eos_token }}\n"
            "{% elif message['role'] == 'system' %}\n{{ '\n' + message['content'] + eos_token }}\n"
            "{% elif message['role'] == 'assistant' %}\n{{ '\n'  + message['content'] + eos_token }}\n"
            "{% endif %}\n"
            "{% if loop.last and add_generation_prompt %}\n{{ '' }}\n{% endif %}\n"
            "{% endfor %}"
        )
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    def _map(ex):
        messages = ex["messages"]
        # Produce a single text sample using the chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    columns = list(dataset["train"].features)
    dataset = dataset.map(_map, remove_columns=columns, desc="Applying chat template")

    # Tokenizer housekeeping
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if getattr(tokenizer, "model_max_length", 2048) > 100_000:
        tokenizer.model_max_length = 2048

    return dataset


def load_model_and_tokenizer(base_model_path: str, tokenizer_path: str, cache_dir: str, hf_token: str, quant_cfg: dict):
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    if hf_token:
        os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token

    tokenizer_src = tokenizer_path if tokenizer_path else base_model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_src,
        cache_dir=cache_dir,
        token=hf_token if hf_token else None,
        trust_remote_code=True,
    )

    # Only create quantization config if load_in_4bit is True
    load_in_4bit = bool(quant_cfg.get("load_in_4bit", False))
    bnb_config = None
    
    if load_in_4bit:
        # Handle None values properly by using 'or' for defaults
        quant_type = quant_cfg.get("bnb_4bit_quant_type") or "nf4"
        compute_dtype_str = quant_cfg.get("bnb_4bit_compute_dtype") or "bfloat16"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quant_type),
            bnb_4bit_compute_dtype=getattr(torch, str(compute_dtype_str))
        )

    # Try to use flash_attention_2 for better packing support
    model_kwargs = {
        "cache_dir": cache_dir,
        "token": hf_token if hf_token else None,
        "device_map": "auto",
        "torch_dtype": "auto",
        "use_cache": False,
        "quantization_config": bnb_config,
        "trust_remote_code": True,
    }
    
    # Try with flash attention first (recommended for packing)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            attn_implementation="flash_attention_2",
            **model_kwargs
        )
        print("✓ Using flash_attention_2 (recommended for packing)")
    except Exception as e:
        print(f"⚠ Flash attention not available ({e}), falling back to default attention")
        print("  Warning: Packing may cause cross-contamination between batches without flash attention")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            **model_kwargs
        )

    return model, tokenizer


def make_lora_config(peft_cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=int(peft_cfg.get("lora_r", 64)),
        lora_alpha=int(peft_cfg.get("lora_alpha", 16)),
        lora_dropout=float(peft_cfg.get("lora_dropout", 0.1)),
        bias=str(peft_cfg.get("bias", "none")),
        task_type="CAUSAL_LM",
        target_modules=list(peft_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]))
    )


def auto_optimize_validation_size(total_dataset_size: int) -> dict:
    """
    Automatically determine optimal validation split for LLM personality fine-tuning.
    
    Philosophy for personality fine-tuning:
    - Need stable signal to measure personality trait adherence
    - Smaller datasets need higher percentage to ensure diversity in validation
    - Larger datasets can use smaller percentage while maintaining absolute size
    - Target validation set size: 300-1000 samples for stable personality metrics
    
    Returns dict with val_size (percentage) and rationale.
    """
    recommendations = {}
    
    if total_dataset_size < 500:
        # Very small - use 25% to ensure at least ~100 validation samples
        val_size = 0.25
        val_count = int(total_dataset_size * val_size)
        rationale = f"Very small dataset: 25% validation ({val_count} samples) for stable personality metrics"
    elif total_dataset_size < 2000:
        # Small - use 20% for good balance
        val_size = 0.20
        val_count = int(total_dataset_size * val_size)
        rationale = f"Small dataset: 20% validation ({val_count} samples) balances training vs. validation"
    elif total_dataset_size < 5000:
        # Small-medium - 15% gives good validation size
        val_size = 0.15
        val_count = int(total_dataset_size * val_size)
        rationale = f"Small-medium dataset: 15% validation ({val_count} samples)"
    elif total_dataset_size < 10000:
        # Medium - 12% is sufficient
        val_size = 0.12
        val_count = int(total_dataset_size * val_size)
        rationale = f"Medium dataset: 12% validation ({val_count} samples)"
    elif total_dataset_size < 20000:
        # Large - 10% still gives 1000+ samples
        val_size = 0.10
        val_count = int(total_dataset_size * val_size)
        rationale = f"Large dataset: 10% validation ({val_count} samples)"
    else:
        # Very large - 5% is plenty for stable metrics
        val_size = 0.05
        val_count = int(total_dataset_size * val_size)
        rationale = f"Very large dataset: 5% validation ({val_count} samples, strong signal)"
    
    recommendations["val_size"] = val_size
    recommendations["val_count_estimate"] = val_count
    recommendations["rationale"] = rationale
    
    return recommendations


def auto_optimize_lora_params(train_dataset_size: int) -> dict:
    """
    Automatically determine optimal LoRA parameters for personality fine-tuning.
    
    Philosophy:
    - Personality is stylistic, not factual → lower rank sufficient
    - Smaller datasets need lower rank to avoid overfitting
    - Larger datasets can use higher rank for more capacity
    - Alpha should be ~2× rank for good adaptation strength
    
    Returns dict with lora_r, lora_alpha, rationale.
    """
    recommendations = {}
    
    if train_dataset_size < 1000:
        # Very small - low rank to prevent overfitting
        lora_r = 8
        lora_alpha = 16
        rationale = f"Very small dataset: rank=8, alpha=16 (prevent overfitting)"
    elif train_dataset_size < 3000:
        # Small-medium - moderate rank
        lora_r = 16
        lora_alpha = 32
        rationale = f"Small-medium dataset: rank=16, alpha=32 (balance capacity & generalization)"
    elif train_dataset_size < 10000:
        # Medium - standard rank for personality
        lora_r = 32
        lora_alpha = 64
        rationale = f"Medium dataset: rank=32, alpha=64 (good capacity for personality)"
    else:
        # Large - can afford higher rank
        lora_r = 32  # Still keep 32, personality doesn't need 64
        lora_alpha = 64
        rationale = f"Large dataset: rank=32, alpha=64 (optimal for personality fine-tuning)"
    
    recommendations["lora_r"] = lora_r
    recommendations["lora_alpha"] = lora_alpha
    recommendations["lora_dropout"] = 0.1  # Standard
    recommendations["rationale"] = rationale
    
    return recommendations


def auto_optimize_batch_config(model_size_b: float = None) -> dict:
    """
    Automatically determine optimal batch size and gradient accumulation.
    
    Philosophy:
    - Maximize GPU utilization while fitting in memory
    - A100 80GB can handle larger batches
    - Maintain effective batch size of 64 (good for stability)
    - Larger per_device_batch = faster training
    
    Args:
        model_size_b: Model size in billions of parameters (e.g., 4.0 for 4B model)
    
    Returns dict with per_device_train_batch_size, gradient_accumulation_steps.
    """
    recommendations = {}
    target_effective_batch = 64  # Good default for personality fine-tuning
    
    # If model size not provided, use conservative defaults
    if model_size_b is None:
        model_size_b = 7.0  # Assume 7B if unknown
    
    if model_size_b <= 4:
        # 4B or smaller - can use larger batches
        per_device_batch = 8
        grad_accum = target_effective_batch // per_device_batch  # 8
        rationale = f"4B model: batch_size=8, grad_accum={grad_accum} (fast training)"
    elif model_size_b <= 7:
        # 7B - moderate batches
        per_device_batch = 4
        grad_accum = target_effective_batch // per_device_batch  # 16
        rationale = f"7B model: batch_size=4, grad_accum={grad_accum} (balanced)"
    elif model_size_b <= 14:
        # 14B - smaller batches
        per_device_batch = 2
        grad_accum = target_effective_batch // per_device_batch  # 32
        rationale = f"14B model: batch_size=2, grad_accum={grad_accum} (memory-safe)"
    else:
        # >14B - very small batches
        per_device_batch = 1
        grad_accum = target_effective_batch  # 64
        rationale = f">14B model: batch_size=1, grad_accum={grad_accum} (large model)"
    
    recommendations["per_device_train_batch_size"] = per_device_batch
    recommendations["per_device_eval_batch_size"] = per_device_batch
    recommendations["gradient_accumulation_steps"] = grad_accum
    recommendations["effective_batch_size"] = target_effective_batch
    recommendations["rationale"] = rationale
    
    return recommendations


def auto_optimize_num_epochs(train_dataset_size: int) -> dict:
    """
    Automatically determine optimal number of training epochs for LLM personality fine-tuning.
    
    Philosophy for personality fine-tuning:
    - Personality is about style/tone, not factual knowledge - needs repetition
    - Small datasets need many epochs to internalize patterns
    - Large datasets have more diversity, need fewer epochs
    - Balance: enough epochs to learn personality, not so many to overfit facts
    
    Returns dict with num_epochs and rationale.
    """
    recommendations = {}
    
    if train_dataset_size < 500:
        # Very small - need many epochs to see patterns
        num_epochs = 80
        rationale = f"Very small dataset: {num_epochs} epochs needed for personality to emerge through repetition"
    elif train_dataset_size < 1000:
        # Small - substantial epochs needed
        num_epochs = 50
        rationale = f"Small dataset: {num_epochs} epochs to internalize personality traits"
    elif train_dataset_size < 3000:
        # Small-medium - moderate-high epochs
        num_epochs = 30
        rationale = f"Small-medium dataset: {num_epochs} epochs for good personality coverage"
    elif train_dataset_size < 5000:
        # Medium-small - moderate epochs
        num_epochs = 20
        rationale = f"Medium-small dataset: {num_epochs} epochs balances learning and overfitting"
    elif train_dataset_size < 10000:
        # Medium - standard epochs
        num_epochs = 15
        rationale = f"Medium dataset: {num_epochs} epochs provides sufficient exposure"
    elif train_dataset_size < 30000:
        # Large - fewer epochs needed
        num_epochs = 10
        rationale = f"Large dataset: {num_epochs} epochs (high diversity reduces repetition needs)"
    else:
        # Very large - minimal epochs
        num_epochs = 5
        rationale = f"Very large dataset: {num_epochs} epochs (excellent diversity, quick learning)"
    
    recommendations["num_epochs"] = num_epochs
    recommendations["rationale"] = rationale
    
    return recommendations


def auto_optimize_training_schedule(train_dataset_size: int, num_epochs: int) -> dict:
    """
    Automatically determine optimal training schedule based on dataset size.
    
    Philosophy:
    - Small datasets (<1k): Evaluate frequently (multiple times per epoch) to catch overfitting early
    - Medium datasets (1k-10k): Evaluate once per epoch for good balance
    - Large datasets (>10k): Can evaluate less frequently to save time
    
    - Early stopping patience should allow enough time to escape local minima
    - Smaller datasets need more patience (in epochs) due to noise
    - Larger datasets can use less patience as signal is stronger
    
    Returns dict with recommended values and rationale.
    """
    recommendations = {}
    
    # Determine eval frequency based on dataset size
    if train_dataset_size < 1000:
        # Very small dataset - evaluate multiple times per epoch to catch overfitting
        eval_steps_per_epoch = 4
        patience_epochs = 15  # More patience due to validation noise
        rationale = "Small dataset (<1k): Frequent evaluation (4x/epoch) to catch overfitting, high patience (15 epochs) for noisy validation"
    elif train_dataset_size < 5000:
        # Small-medium dataset - evaluate 2x per epoch
        eval_steps_per_epoch = 2
        patience_epochs = 15  # Higher patience for personality fine-tuning (non-linear learning)
        rationale = "Small-medium dataset (1k-5k): Moderate evaluation (2x/epoch), patience = 15 epochs (75% of typical 20 epoch run)"
    elif train_dataset_size < 10000:
        # Medium dataset - evaluate once per epoch
        eval_steps_per_epoch = 1
        patience_epochs = 10
        rationale = "Medium dataset (5k-10k): Evaluate once per epoch, patience = 10 epochs"
    elif train_dataset_size < 50000:
        # Large dataset - can evaluate less frequently
        eval_steps_per_epoch = 1
        patience_epochs = 8
        rationale = "Large dataset (10k-50k): Once per epoch, patience = 8 epochs (strong signal)"
    else:
        # Very large dataset - minimal evaluation needed
        eval_steps_per_epoch = 1
        patience_epochs = 5
        rationale = "Very large dataset (>50k): Once per epoch, patience = 5 epochs (very strong signal)"
    
    # Adjust patience based on total epochs
    if num_epochs < patience_epochs:
        # If training for fewer epochs than patience, reduce patience
        patience_epochs = max(3, num_epochs // 2)
        rationale += f" [Adjusted patience to {patience_epochs} based on total epochs={num_epochs}]"
    
    recommendations["eval_steps_per_epoch"] = eval_steps_per_epoch
    recommendations["early_stopping_patience"] = patience_epochs
    recommendations["rationale"] = rationale
    
    return recommendations


def make_sft_config(training_cfg: dict, output_dir: str, tokenizer: AutoTokenizer, train_dataset_size: int = None) -> SFTConfig:
    """Build an SFTConfig compatible with the installed TRL version.

    Notes:
    - Some TRL/Transformers versions do not accept certain arguments (e.g., max_seq_length, dataset_text_field,
      gradient_checkpointing_kwargs). We introspect SFTConfig.__init__ and only pass supported keys.
    - For controlling sequence length, prefer setting tokenizer.model_max_length upstream when 'max_seq_length'
      is not supported by this TRL version.
    - If eval_steps_per_epoch is specified and train_dataset_size is provided, eval_steps will be auto-calculated.
    """
    # Auto-optimize training schedule if dataset size is provided
    auto_mode = False
    if train_dataset_size is not None:
        num_epochs = int(training_cfg.get("num_train_epochs", 1))
        auto_recommendations = auto_optimize_training_schedule(train_dataset_size, num_epochs)
        
        # Use auto-recommendations if not explicitly set in config
        # If eval_steps_per_epoch is "auto" or not set, use recommended value
        eval_steps_per_epoch_cfg = training_cfg.get("eval_steps_per_epoch", "auto")
        if eval_steps_per_epoch_cfg == "auto" or str(eval_steps_per_epoch_cfg).lower() == "auto":
            auto_mode = True
            print(f"\n{'='*60}")
            print(f"📊 AUTO-OPTIMIZATION (train_size={train_dataset_size})")
            print(f"{'='*60}")
            print(f"Recommendation: {auto_recommendations['rationale']}")
            print(f"  • eval_steps_per_epoch: {auto_recommendations['eval_steps_per_epoch']}")
            print(f"  • early_stopping_patience: {auto_recommendations['early_stopping_patience']} epochs")
            print(f"{'='*60}\n")
            
            # Override config values with auto-recommendations
            training_cfg = dict(training_cfg)  # Make a copy
            training_cfg["eval_steps_per_epoch"] = auto_recommendations["eval_steps_per_epoch"]
            
            # Also auto-set patience if it's set to "auto"
            patience_cfg = training_cfg.get("early_stopping_patience", "auto")
            if patience_cfg == "auto" or str(patience_cfg).lower() == "auto":
                training_cfg["early_stopping_patience"] = auto_recommendations["early_stopping_patience"]
    
    # Calculate eval_steps if eval_steps_per_epoch is specified
    eval_steps = int(training_cfg.get("eval_steps", 100))
    save_steps = int(training_cfg.get("save_steps", 100))
    
    if "eval_steps_per_epoch" in training_cfg and train_dataset_size is not None:
        eval_steps_per_epoch_val = training_cfg["eval_steps_per_epoch"]
        # Skip if it's the string "auto" (shouldn't happen after override, but just in case)
        if str(eval_steps_per_epoch_val).lower() == "auto":
            eval_steps_per_epoch = 1  # Fallback
        else:
            eval_steps_per_epoch = int(eval_steps_per_epoch_val)
        per_device_batch = int(training_cfg.get("per_device_train_batch_size", 1))
        grad_accum = int(training_cfg.get("gradient_accumulation_steps", 1))
        effective_batch = per_device_batch * grad_accum
        steps_per_epoch = max(1, train_dataset_size // effective_batch)
        eval_steps = max(1, steps_per_epoch // eval_steps_per_epoch)
        
        # Auto-set save_steps to match eval_steps (save at same frequency as eval)
        if "save_steps" not in training_cfg or training_cfg.get("save_steps") == 100:  # 100 is default
            save_steps = eval_steps
        
        print(f"Auto-calculated eval_steps: {eval_steps} (evaluating {eval_steps_per_epoch}x per epoch of {steps_per_epoch} steps)")
        print(f"Auto-calculated save_steps: {save_steps} (saving {eval_steps_per_epoch}x per epoch)")
        
        # When eval_steps_per_epoch = 1, early_stopping_patience is in epochs
        if eval_steps_per_epoch == 1:
            patience_check = training_cfg.get("early_stopping_patience", 0)
            if patience_check and str(patience_check).lower() != "auto":
                try:
                    patience_int = int(patience_check)
                    if patience_int > 0:
                        print(f"Early stopping: patience = {patience_int} epochs (eval happens 1x per epoch)")
                except (ValueError, TypeError):
                    pass
    
    # Auto-match save_total_limit to early_stopping_patience
    # This ensures we keep all checkpoints from the patience window
    patience_val = training_cfg.get("early_stopping_patience", 0)
    # Handle both numeric and "auto" values
    if str(patience_val).lower() == "auto":
        patience = 0  # Will be set by auto-optimization
    else:
        patience = int(patience_val) if patience_val else 0
    
    if patience > 0:
        save_total_limit = patience
        print(f"Auto-set save_total_limit = {save_total_limit} (matches early_stopping_patience)")
    else:
        save_total_limit = int(training_cfg.get("save_total_limit", 2))
    
    base_kwargs = {
        "bf16": bool(training_cfg.get("bf16", True)),
        "do_eval": True,
        "eval_strategy": "steps" if eval_steps > 0 else "no",
        "eval_steps": eval_steps,
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 64)),
        "gradient_checkpointing": bool(training_cfg.get("gradient_checkpointing", True)),
        "learning_rate": float(training_cfg.get("learning_rate", 2e-5)),
        "log_level": "info",
        "logging_steps": int(training_cfg.get("logging_steps", 1)),
        "logging_strategy": "steps",
        "lr_scheduler_type": str(training_cfg.get("lr_scheduler_type", "cosine")),
        "max_steps": int(training_cfg.get("max_steps", -1)),
        "num_train_epochs": int(training_cfg.get("num_train_epochs", 1)),
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "per_device_eval_batch_size": int(training_cfg.get("per_device_eval_batch_size", 1)),
        "per_device_train_batch_size": int(training_cfg.get("per_device_train_batch_size", 1)),
        "report_to": list(training_cfg.get("report_to", [])),
        "save_strategy": "steps" if save_steps > 0 else "no",
        "save_steps": save_steps,  # Auto-calculated to match eval_steps
        "save_total_limit": save_total_limit,  # Auto-set to match patience
        "load_best_model_at_end": bool(training_cfg.get("load_best_model_at_end", True)),
        "metric_for_best_model": training_cfg.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": bool(training_cfg.get("greater_is_better", False)),
        "seed": int(training_cfg.get("seed", 42)),
        # TRL (incl. v0.23) supports packing in SFTConfig
        "packing": bool(training_cfg.get("packing", True)),
    }

    optional_kwargs = {
        # Some TRL versions accept this; Transformers may also need to support it
        "gradient_checkpointing_kwargs": {"use_reentrant": bool(training_cfg.get("gradient_checkpointing_reentrant", False))},
        # Older TRL versions accepted dataset_text_field; newer ones auto-detect 'text' or 'messages'
        "dataset_text_field": "text",
        # Do not include max_seq_length here; manage via tokenizer.model_max_length instead
    }

    sig = inspect.signature(SFTConfig.__init__)
    allowed = set(sig.parameters.keys())
    cfg_kwargs = {}
    for k, v in {**base_kwargs, **optional_kwargs}.items():
        if k in allowed:
            cfg_kwargs[k] = v

    # Ensure evaluation strategy is set with the correct key for this TRL version
    eval_val = "steps" if int(training_cfg.get("eval_steps", 0)) > 0 else "no"
    if "evaluation_strategy" in allowed:
        cfg_kwargs["evaluation_strategy"] = eval_val
    elif "eval_strategy" in allowed and "eval_strategy" not in cfg_kwargs:
        cfg_kwargs["eval_strategy"] = eval_val

    return SFTConfig(**cfg_kwargs)


def save_configs_yaml_json(output_dir: str, cfg: Dict[str, Any], sft_config: SFTConfig, lora_config=None):
    # Save the input YAML for reproducibility
    with open(os.path.join(output_dir, "run_config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    # Save trainer configs
    import json as _json
    with open(os.path.join(output_dir, 'sft_config.json'), 'w') as f:
        _json.dump(sft_config.to_dict(), f, default=list, indent=2)
    if lora_config is not None:
        with open(os.path.join(output_dir, 'peft_config.json'), 'w') as f:
            _json.dump(lora_config.to_dict(), f, default=list, indent=2)


def main():
    parser = argparse.ArgumentParser(description="SFT trainer for personality trait subset (supports full FT and QLoRA)")
    parser.add_argument("--config", type=str, default="personality_sft.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    random.seed(seed)

    paths = cfg.get("paths", {})
    base_model_path = paths.get("base_model_path")
    tokenizer_path = paths.get("tokenizer_path")
    input_json = paths.get("input_json")
    output_dir = paths.get("output_dir", "./outputs/personality-lora")
    cache_dir = paths.get("cache_dir", "./hf_cache")
    hf_token = cfg.get("hf", {}).get("token", "")

    ensure_dir(output_dir)
    ensure_dir(cache_dir)

    quant_cfg = cfg.get("quantization", {})
    peft_cfg = cfg.get("peft", {})
    data_cfg = cfg.get("data", {})
    training_cfg = cfg.get("training", {})

    if not base_model_path:
        raise ValueError("paths.base_model_path is required in config")
    if not input_json or not os.path.isfile(input_json):
        raise ValueError("paths.input_json must point to an existing JSON file of your personality trait subset")

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(
        base_model_path=base_model_path,
        tokenizer_path=tokenizer_path,
        cache_dir=cache_dir,
        hf_token=hf_token,
        quant_cfg=quant_cfg,
    )

    # Optionally control tokenizer max length via config for TRL versions without 'max_seq_length'
    try:
        _max_len = int(training_cfg.get("max_seq_length", 0))
    except Exception:
        _max_len = 0
    if _max_len and _max_len > 0:
        try:
            tokenizer.model_max_length = _max_len
        except Exception:
            pass

    # Auto-optimization: First, count total dataset size and detect model size
    print("\n" + "="*70)
    print("🔍 ANALYZING DATASET & MODEL FOR AUTO-OPTIMIZATION")
    print("="*70)
    
    # Quick count of dataset size
    data = []
    with open(input_json, "r") as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError(f"Empty file: {input_json}")
        f.seek(0)
        try:
            content = f.read()
            parsed = json.loads(content)
            if isinstance(parsed, list):
                data = parsed
            else:
                data = [parsed]
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    total_dataset_size = len(data)
    print(f"📊 Total dataset size: {total_dataset_size} samples")
    
    # Detect model size from base_model_path (e.g., "Qwen3-4B" -> 4.0)
    model_size_b = None
    try:
        import re
        model_name = os.path.basename(base_model_path.rstrip('/'))
        # Match patterns like "4B", "7B", "14B"
        match = re.search(r'(\d+\.?\d*)B', model_name, re.IGNORECASE)
        if match:
            model_size_b = float(match.group(1))
            print(f"📏 Detected model size: {model_size_b}B parameters")
    except Exception:
        pass
    
    # Auto-optimize validation size if set to "auto"
    val_size_raw = data_cfg.get("val_size", 1000)
    if str(val_size_raw).lower() == "auto":
        val_recommendations = auto_optimize_validation_size(total_dataset_size)
        val_size = val_recommendations["val_size"]
        print(f"\n✅ AUTO VALIDATION SIZE:")
        print(f"   {val_recommendations['rationale']}")
        print(f"   → Using val_size = {val_size} ({val_size*100:.0f}%)")
        data_cfg = dict(data_cfg)
        data_cfg["val_size"] = val_size
    elif isinstance(val_size_raw, float):
        val_size = val_size_raw
    else:
        val_size = int(val_size_raw)
    
    # Auto-optimize LoRA parameters if set to "auto"
    estimated_train_size = int(total_dataset_size * (1 - val_size))
    
    lora_r_raw = peft_cfg.get("lora_r", 16)
    lora_alpha_raw = peft_cfg.get("lora_alpha", 32)
    if str(lora_r_raw).lower() == "auto" or str(lora_alpha_raw).lower() == "auto":
        lora_recommendations = auto_optimize_lora_params(estimated_train_size)
        print(f"\n✅ AUTO LORA PARAMETERS:")
        print(f"   {lora_recommendations['rationale']}")
        print(f"   → lora_r = {lora_recommendations['lora_r']}")
        print(f"   → lora_alpha = {lora_recommendations['lora_alpha']}")
        print(f"   → lora_dropout = {lora_recommendations['lora_dropout']}")
        peft_cfg = dict(peft_cfg)
        peft_cfg["lora_r"] = lora_recommendations["lora_r"]
        peft_cfg["lora_alpha"] = lora_recommendations["lora_alpha"]
        peft_cfg["lora_dropout"] = lora_recommendations["lora_dropout"]
    
    # Auto-optimize batch size and gradient accumulation if set to "auto"
    batch_size_raw = training_cfg.get("per_device_train_batch_size", 1)
    grad_accum_raw = training_cfg.get("gradient_accumulation_steps", 64)
    if str(batch_size_raw).lower() == "auto" or str(grad_accum_raw).lower() == "auto":
        batch_recommendations = auto_optimize_batch_config(model_size_b)
        print(f"\n✅ AUTO BATCH CONFIGURATION:")
        print(f"   {batch_recommendations['rationale']}")
        print(f"   → per_device_train_batch_size = {batch_recommendations['per_device_train_batch_size']}")
        print(f"   → gradient_accumulation_steps = {batch_recommendations['gradient_accumulation_steps']}")
        print(f"   → effective_batch_size = {batch_recommendations['effective_batch_size']}")
        training_cfg = dict(training_cfg)
        training_cfg["per_device_train_batch_size"] = batch_recommendations["per_device_train_batch_size"]
        training_cfg["per_device_eval_batch_size"] = batch_recommendations["per_device_eval_batch_size"]
        training_cfg["gradient_accumulation_steps"] = batch_recommendations["gradient_accumulation_steps"]
    
    # Auto-optimize num_train_epochs if set to "auto"
    num_epochs_raw = training_cfg.get("num_train_epochs", 1)
    if str(num_epochs_raw).lower() == "auto":
        epoch_recommendations = auto_optimize_num_epochs(estimated_train_size)
        num_train_epochs = epoch_recommendations["num_epochs"]
        print(f"\n✅ AUTO NUM EPOCHS:")
        print(f"   {epoch_recommendations['rationale']}")
        print(f"   → Using num_train_epochs = {num_train_epochs}")
        training_cfg = dict(training_cfg)
        training_cfg["num_train_epochs"] = num_train_epochs
    
    # Auto-optimize logging_steps if too fine-grained
    logging_steps_raw = training_cfg.get("logging_steps", 1)
    if logging_steps_raw == 1 and estimated_train_size > 1000:
        # For datasets > 1000, log every 10 steps for cleaner curves
        recommended_logging = 10
        print(f"\n💡 AUTO LOGGING ADJUSTMENT:")
        print(f"   logging_steps: 1 → {recommended_logging} (cleaner curves, less overhead)")
        training_cfg = dict(training_cfg)
        training_cfg["logging_steps"] = recommended_logging
    
    print("="*70 + "\n")
    
    # Build dataset
    dataset = json_to_hf_dataset(
        json_path=input_json,
        val_size=val_size,
        add_system=bool(data_cfg.get("add_system_message", True)),
        system_message=str(data_cfg.get("system_message", "Reply concisely within 20 words")),
        seed=seed,
    )

    dataset = apply_template_and_tokenize(dataset, tokenizer)

    # Determine if using PEFT (LoRA) or full fine-tuning
    use_peft = peft_cfg.get("use_peft", True)  # Default to True for backward compatibility
    
    # LoRA config (only if using PEFT)
    lora_config = make_lora_config(peft_cfg) if use_peft else None
    
    # Get train dataset size for eval_steps calculation
    train_dataset_size = len(dataset["train"])
    sft_config = make_sft_config(training_cfg, output_dir, tokenizer, train_dataset_size)

    # Trainer
    callbacks = []
    patience = training_cfg.get("early_stopping_patience", 0)
    if isinstance(patience, int) and patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,  # None for full fine-tuning, LoraConfig for PEFT
        callbacks=callbacks,
    )

    train_result = trainer.train()

    # Save outputs
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_configs_yaml_json(output_dir, cfg, sft_config, lora_config)
    
    # Explicitly save trainer state for metrics and plotting
    trainer_state_path = os.path.join(output_dir, "trainer_state.json")
    trainer.state.save_to_json(trainer_state_path)
    print(f"✅ Saved trainer state to: {trainer_state_path}")

    print(f"\nTraining complete. Adapters and configs saved to: {output_dir}\n")


if __name__ == "__main__":
    main()

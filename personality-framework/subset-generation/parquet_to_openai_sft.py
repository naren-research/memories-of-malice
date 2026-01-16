#!/usr/bin/env python3
"""
Convert filtered OCEAN parquet subsets to OpenAI supervised fine-tuning JSONL format.

This script reads parquet files containing dialogue data and converts them to the
JSONL format required by OpenAI's fine-tuning API.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

logger = logging.getLogger("parquet_to_openai_sft")


def dialogue_to_messages(
    dialogue: List[str],
    system_prompt: Optional[str] = None,
    assistant_is_speaker1: bool = True,
) -> List[Dict[str, str]]:
    """
    Convert a dialogue list to OpenAI messages format.
    
    Args:
        dialogue: List of dialogue turns (alternating between speakers)
        system_prompt: Optional system prompt to prepend
        assistant_is_speaker1: If True, speaker 1 (even indices) is assistant,
                               otherwise speaker 2 (odd indices) is assistant
    
    Returns:
        List of message dicts with 'role' and 'content' keys.
        Ensures conversation always starts with 'user' and ends with 'assistant'.
    """
    messages = []
    
    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Convert dialogue turns to user/assistant messages
    # Assign roles based on which speaker should be the assistant
    for i, turn in enumerate(dialogue):
        if not turn or not turn.strip():
            continue
        
        # Determine role based on speaker assignment
        is_speaker1 = (i % 2 == 0)
        if assistant_is_speaker1:
            role = "assistant" if is_speaker1 else "user"
        else:
            role = "user" if is_speaker1 else "assistant"
        
        messages.append({"role": role, "content": turn.strip()})
    
    # Ensure conversation starts with user and ends with assistant
    # If it starts with assistant, swap all roles
    conversation = [m for m in messages if m['role'] != 'system']
    if conversation and conversation[0]['role'] == 'assistant':
        for msg in messages:
            if msg['role'] == 'user':
                msg['role'] = 'assistant'
            elif msg['role'] == 'assistant':
                msg['role'] = 'user'
    
    return messages


def convert_parquet_to_multiple_jsonl(
    input_path: Path,
    output_configs: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
    min_turns: int = 2,
    max_turns: Optional[int] = None,
    batch_size: int = 10000,
    target_trait: str = "agreeableness",
) -> None:
    """
    Convert parquet file to multiple OpenAI JSONL format files (one per split).
    
    Args:
        input_path: Path to input parquet file
        output_configs: List of dicts with 'split' and 'output_path' keys
        system_prompt: Optional system prompt to add to each example
        min_turns: Minimum number of dialogue turns to include
        max_turns: Maximum number of dialogue turns to include (None = no limit)
        batch_size: Number of rows to process at once
        target_trait: Personality trait to use for selecting assistant speaker
    """
    logger.info("Reading parquet file: %s", input_path)
    parquet_file = pq.ParquetFile(input_path)
    
    # Open all output files
    output_files = {}
    split_stats = {}
    
    for config in output_configs:
        split_name = config["split"]
        output_path = config["output_path"]
        output_files[split_name] = open(output_path, "w", encoding="utf-8")
        split_stats[split_name] = {"written": 0, "skipped": 0}
        logger.info("Will write %s split to: %s", split_name, output_path)
    
    total_records = 0
    total_skipped = 0
    
    try:
        for batch in tqdm(
            parquet_file.iter_batches(batch_size=batch_size),
            desc="Converting to JSONL",
            unit="batch",
        ):
            records = batch.to_pylist()
            total_records += len(records)
            
            for record in records:
                record_split = record.get("split")
                
                # Skip if split not in our output configs
                if record_split not in output_files:
                    total_skipped += 1
                    continue
                
                dialogue = record.get("dialogue", [])
                
                # Filter by dialogue length
                if len(dialogue) < min_turns:
                    split_stats[record_split]["skipped"] += 1
                    total_skipped += 1
                    continue
                
                if max_turns and len(dialogue) > max_turns:
                    split_stats[record_split]["skipped"] += 1
                    total_skipped += 1
                    continue
                
                # Determine which speaker should be the assistant based on target trait
                scores1 = record.get("journal_entry1_scores", {})
                scores2 = record.get("journal_entry2_scores", {})
                
                trait_score1 = scores1.get(target_trait, 0.5)
                trait_score2 = scores2.get(target_trait, 0.5)
                
                # Assign the speaker with higher trait score as assistant
                assistant_is_speaker1 = trait_score1 >= trait_score2
                
                # Convert to OpenAI format
                messages = dialogue_to_messages(dialogue, system_prompt, assistant_is_speaker1)
                
                # Skip if no valid messages
                if len(messages) < 2:
                    split_stats[record_split]["skipped"] += 1
                    total_skipped += 1
                    continue
                
                # Create JSONL entry
                entry = {"messages": messages}
                
                # Add UUID if available
                if "uuid" in record:
                    entry["uuid"] = record["uuid"]
                
                # Write to appropriate file
                output_files[record_split].write(json.dumps(entry) + "\n")
                split_stats[record_split]["written"] += 1
    
    finally:
        # Close all output files
        for f in output_files.values():
            f.close()
    
    # Log statistics
    logger.info(
        "Conversion complete: %d total records, %d skipped",
        total_records,
        total_skipped,
    )
    for split_name, stats in split_stats.items():
        logger.info(
            "  %s: %d written, %d skipped",
            split_name,
            stats["written"],
            stats["skipped"],
        )


def load_system_prompts(prompts_path: Path) -> Dict[str, Dict[str, str]]:
    """Load system prompts from JSON file."""
    with open(prompts_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def get_system_prompt(
    system_prompts: Dict[str, Dict[str, str]],
    trait: str,
    mode: str,
) -> str:
    """
    Get system prompt for a specific trait and mode.
    
    Args:
        system_prompts: Dict of trait -> {mode -> prompt}
        trait: Trait name (e.g., 'openness', 'conscientiousness')
        mode: Mode ('high' or 'low')
    
    Returns:
        System prompt string
    """
    if trait not in system_prompts:
        raise ValueError(f"Unknown trait '{trait}'. Available: {list(system_prompts.keys())}")
    if mode not in system_prompts[trait]:
        raise ValueError(f"Unknown mode '{mode}' for trait '{trait}'. Available: {list(system_prompts[trait].keys())}")
    return system_prompts[trait][mode]


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}
    
    if "input" not in config:
        raise ValueError("Config must include 'input' (path to parquet file)")
    if "output_splits" not in config:
        raise ValueError("Config must include 'output_splits' with split definitions")
    
    return config


def resolve_path(path_like: Any, base_dir: Path) -> Path:
    """Resolve a path relative to base_dir if not absolute."""
    path = Path(str(path_like))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert OCEAN-scored parquet to OpenAI supervised fine-tuning JSONL format."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file",
    )
    
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    
    # Load config
    config = load_config(config_path)
    
    # Setup logging
    log_level_name = str(config.get("log_level", "INFO")).upper()
    level_map = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    if log_level_name not in level_map:
        raise ValueError(f"Unknown log_level '{log_level_name}'")
    
    logging.basicConfig(
        level=level_map[log_level_name],
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger.info("Loaded configuration from %s", config_path)
    
    # Resolve paths
    config_dir = config_path.parent
    base_dir_value = config.get("base_dir")
    if base_dir_value:
        base_dir = Path(base_dir_value)
        if not base_dir.is_absolute():
            base_dir = (config_dir / base_dir).resolve()
    else:
        base_dir = config_dir
    
    input_path = resolve_path(config["input"], base_dir)
    
    # Handle system prompt: can be direct string, or reference to JSON file
    system_prompt = None
    if "system_prompt" in config:
        system_prompt = config["system_prompt"]
    elif "system_prompt_config" in config:
        # Load from JSON file
        prompt_config = config["system_prompt_config"]
        prompts_file = resolve_path(prompt_config["file"], base_dir)
        trait = prompt_config.get("trait")
        mode = prompt_config.get("mode", "high")
        
        if not trait:
            raise ValueError("system_prompt_config must specify 'trait'")
        
        logger.info("Loading system prompts from %s", prompts_file)
        system_prompts = load_system_prompts(prompts_file)
        system_prompt = get_system_prompt(system_prompts, trait, mode)
        logger.info("Using %s-%s system prompt", trait, mode)
    
    min_turns = int(config.get("min_turns", 2))
    max_turns = config.get("max_turns")
    if max_turns is not None:
        max_turns = int(max_turns)
    batch_size = int(config.get("batch_size", 10000))
    target_trait = config.get("target_trait", "agreeableness")
    
    # Parse output configuration
    output_splits = config.get("output_splits", {})
    if not output_splits:
        raise ValueError("Config must include 'output_splits' with split definitions")
    
    # Build output configs
    output_configs = []
    for split_name, split_config in output_splits.items():
        if isinstance(split_config, str):
            # Simple case: just output path
            output_path = resolve_path(split_config, base_dir)
        elif isinstance(split_config, dict):
            # Complex case: dict with output path
            output_path = resolve_path(split_config["output"], base_dir)
        else:
            raise ValueError(f"Invalid output config for split '{split_name}'")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_configs.append({
            "split": split_name,
            "output_path": output_path,
        })
    
    # Run conversion
    convert_parquet_to_multiple_jsonl(
        input_path=input_path,
        output_configs=output_configs,
        system_prompt=system_prompt,
        min_turns=min_turns,
        max_turns=max_turns,
        batch_size=batch_size,
        target_trait=target_trait,
    )


if __name__ == "__main__":
    main()

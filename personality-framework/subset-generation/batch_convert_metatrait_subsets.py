#!/usr/bin/env python3
"""
Batch convert all metatrait-based parquet subsets to OpenAI JSONL format.

This script processes all 9 metatrait subsets and generates training JSONL files
with appropriate system prompts for each subset.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pyarrow.parquet as pq
from tqdm import tqdm
import argparse

logger = logging.getLogger("batch_convert_metatrait")


def load_system_prompts(prompts_path: Path) -> Dict[str, str]:
    """Load system prompts from JSON file."""
    with open(prompts_path, "r", encoding="utf-8") as fp:
        prompts = json.load(fp)
    
    # Flatten nested structure for metatraits
    flat_prompts = {}
    for key, value in prompts.items():
        if isinstance(value, dict):
            # Old format: trait -> {high/low -> prompt}
            for subkey, prompt in value.items():
                flat_prompts[f"{key}_{subkey}"] = prompt
        else:
            # New format: direct key -> prompt
            flat_prompts[key] = value
    
    return flat_prompts


def dialogue_to_messages(
    dialogue: List[str],
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Convert a dialogue list to OpenAI messages format.
    Alternates between user and assistant, always starting with user.
    
    Args:
        dialogue: List of dialogue turns
        system_prompt: Optional system prompt to prepend
    
    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    messages = []
    
    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Convert dialogue turns to user/assistant messages
    # Alternating pattern: user, assistant, user, assistant, ...
    for i, turn in enumerate(dialogue):
        if not turn or not turn.strip():
            continue
        
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": turn.strip()})
    
    # Ensure conversation starts with user and ends with assistant
    conversation = [m for m in messages if m['role'] != 'system']
    if not conversation:
        return messages
    
    # If starts with assistant, swap all roles
    if conversation[0]['role'] == 'assistant':
        for msg in messages:
            if msg['role'] == 'user':
                msg['role'] = 'assistant'
            elif msg['role'] == 'assistant':
                msg['role'] = 'user'
    
    return messages


def convert_parquet_to_jsonl(
    input_path: Path,
    output_path: Path,
    system_prompt: str,
    min_turns: int = 2,
    max_turns: Optional[int] = None,
) -> Dict[str, int]:
    """
    Convert single parquet file to OpenAI JSONL format.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output JSONL file
        system_prompt: System prompt to add to each example
        min_turns: Minimum number of dialogue turns
        max_turns: Maximum number of dialogue turns (None = no limit)
    
    Returns:
        Dictionary with conversion statistics
    """
    logger.info(f"Converting: {input_path.name}")
    
    # Read parquet
    df = pq.read_table(input_path).to_pandas()
    
    stats = {
        "total": len(df),
        "written": 0,
        "skipped_short": 0,
        "skipped_long": 0,
        "skipped_invalid": 0,
    }
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as out_file:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {input_path.stem}"):
            dialogue = row.get("dialogue", [])
            
            # Filter by dialogue length
            if len(dialogue) < min_turns:
                stats["skipped_short"] += 1
                continue
            
            if max_turns and len(dialogue) > max_turns:
                stats["skipped_long"] += 1
                continue
            
            # Convert to OpenAI format
            messages = dialogue_to_messages(dialogue, system_prompt)
            
            # Skip if no valid messages
            if len(messages) < 2:
                stats["skipped_invalid"] += 1
                continue
            
            # Create JSONL entry
            entry = {"messages": messages}
            
            # Add UUID if available
            if "uuid" in row and row["uuid"]:
                entry["uuid"] = row["uuid"]
            
            # Write to file
            out_file.write(json.dumps(entry) + "\n")
            stats["written"] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert metatrait parquet subsets to OpenAI JSONL format"
    )
    parser.add_argument(
        "--subsets-dir",
        type=Path,
        default=Path("./subsets"),
        help="Directory containing parquet subset files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for JSONL files (defaults to subsets-dir/jsonl)"
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=Path("./configs/system_prompts.json"),
        help="Path to system prompts JSON file"
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=2,
        help="Minimum number of dialogue turns"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum number of dialogue turns (None = no limit)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.subsets_dir / "jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load system prompts
    logger.info(f"Loading system prompts from: {args.prompts_file}")
    system_prompts = load_system_prompts(args.prompts_file)
    
    # Define subset configurations
    subset_configs = [
        "stability_high",
        "stability_low",
        "plasticity_high",
        "plasticity_low",
        "stability_neutral",
        "plasticity_neutral",
        "stability_neutral_plasticity_neutral",
        "stability_high_plasticity_high",
        "stability_low_plasticity_low",
        "stability_high_plasticity_low",
        "stability_low_plasticity_high",
    ]
    
    logger.info(f"Processing {len(subset_configs)} subsets...")
    logger.info(f"Input directory: {args.subsets_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Process each subset
    all_stats = {}
    for subset_name in subset_configs:
        input_file = args.subsets_dir / f"{subset_name}.parquet"
        output_file = output_dir / f"{subset_name}.jsonl"
        
        if not input_file.exists():
            logger.warning(f"Skipping {subset_name}: file not found at {input_file}")
            continue
        
        if subset_name not in system_prompts:
            logger.error(f"No system prompt found for '{subset_name}'")
            continue
        
        system_prompt = system_prompts[subset_name]
        
        # Convert
        stats = convert_parquet_to_jsonl(
            input_path=input_file,
            output_path=output_file,
            system_prompt=system_prompt,
            min_turns=args.min_turns,
            max_turns=args.max_turns,
        )
        
        all_stats[subset_name] = stats
        
        logger.info(
            f"  {subset_name}: {stats['written']}/{stats['total']} written "
            f"(skipped: {stats['skipped_short']} short, {stats['skipped_long']} long, "
            f"{stats['skipped_invalid']} invalid)"
        )
    
    # Summary
    print("\n" + "="*80)
    print("CONVERSION SUMMARY")
    print("="*80)
    print(f"{'Subset':<25} {'Total':<10} {'Written':<10} {'Skipped':<10} {'Success %':<10}")
    print("-"*80)
    
    total_all = 0
    written_all = 0
    
    for subset_name, stats in all_stats.items():
        total = stats['total']
        written = stats['written']
        skipped = total - written
        success_pct = (written / total * 100) if total > 0 else 0
        
        print(f"{subset_name:<25} {total:<10,} {written:<10,} {skipped:<10,} {success_pct:<10.1f}")
        
        total_all += total
        written_all += written
    
    print("-"*80)
    success_all = (written_all / total_all * 100) if total_all > 0 else 0
    print(f"{'TOTAL':<25} {total_all:<10,} {written_all:<10,} {total_all-written_all:<10,} {success_all:<10.1f}")
    print("="*80)
    print(f"\n✅ Conversion complete! JSONL files saved to: {output_dir}")


if __name__ == "__main__":
    main()

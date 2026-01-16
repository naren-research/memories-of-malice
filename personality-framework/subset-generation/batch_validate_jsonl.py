#!/usr/bin/env python3
"""
Batch validate all generated JSONL files for OpenAI fine-tuning.

This script validates all JSONL files in a directory to ensure they meet
OpenAI's requirements for supervised fine-tuning.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def validate_message(msg: Dict, line_num: int, msg_idx: int) -> List[str]:
    """Validate a single message."""
    errors = []
    
    if "role" not in msg:
        errors.append(f"Line {line_num}, message {msg_idx}: Missing 'role' key")
    elif msg["role"] not in ["system", "user", "assistant"]:
        errors.append(f"Line {line_num}, message {msg_idx}: Invalid role '{msg['role']}'")
    
    if "content" not in msg:
        errors.append(f"Line {line_num}, message {msg_idx}: Missing 'content' key")
    elif not isinstance(msg["content"], str):
        errors.append(f"Line {line_num}, message {msg_idx}: 'content' must be a string")
    elif not msg["content"].strip():
        errors.append(f"Line {line_num}, message {msg_idx}: 'content' is empty")
    
    return errors


def validate_jsonl_file(file_path: Path) -> Tuple[bool, Dict]:
    """
    Validate a single JSONL file.
    
    Returns:
        Tuple of (is_valid, stats_dict)
    """
    errors = []
    warnings = []
    total_lines = 0
    total_tokens_estimate = 0
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                total_lines += 1
                
                # Check if line is valid JSON
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")
                    continue
                
                # Check for 'messages' key
                if "messages" not in data:
                    errors.append(f"Line {line_num}: Missing 'messages' key")
                    continue
                
                messages = data["messages"]
                if not isinstance(messages, list):
                    errors.append(f"Line {line_num}: 'messages' must be a list")
                    continue
                
                if len(messages) < 2:
                    warnings.append(f"Line {line_num}: Only {len(messages)} message(s)")
                
                # Validate each message
                for msg_idx, msg in enumerate(messages):
                    errors.extend(validate_message(msg, line_num, msg_idx))
                
                # Estimate tokens (rough: 4 chars = 1 token)
                line_tokens = len(line) // 4
                total_tokens_estimate += line_tokens
    
    except FileNotFoundError:
        errors.append(f"File not found: {file_path}")
    except Exception as e:
        errors.append(f"Error reading file: {e}")
    
    # Check minimum examples
    if total_lines < 10:
        warnings.append(f"Only {total_lines} examples (OpenAI requires minimum 10)")
    
    stats = {
        "total_lines": total_lines,
        "total_tokens_estimate": total_tokens_estimate,
        "errors": errors,
        "warnings": warnings,
        "is_valid": len(errors) == 0,
    }
    
    return len(errors) == 0, stats


def print_file_summary(file_path: Path, stats: Dict, verbose: bool = False):
    """Print validation summary for a single file."""
    status = "✅ PASS" if stats["is_valid"] else "❌ FAIL"
    print(f"\n{status} | {file_path.name}")
    print(f"  Examples: {stats['total_lines']:,}")
    print(f"  Est. Tokens: ~{stats['total_tokens_estimate']:,}")
    print(f"  Errors: {len(stats['errors'])}")
    print(f"  Warnings: {len(stats['warnings'])}")
    
    if verbose or not stats["is_valid"]:
        if stats["errors"]:
            print(f"\n  Errors:")
            for error in stats["errors"][:5]:  # Show first 5
                print(f"    - {error}")
            if len(stats["errors"]) > 5:
                print(f"    ... and {len(stats['errors']) - 5} more errors")
        
        if stats["warnings"]:
            print(f"\n  Warnings:")
            for warning in stats["warnings"][:5]:  # Show first 5
                print(f"    - {warning}")
            if len(stats["warnings"]) > 5:
                print(f"    ... and {len(stats['warnings']) - 5} more warnings")


def main():
    parser = argparse.ArgumentParser(
        description="Batch validate JSONL files for OpenAI fine-tuning"
    )
    parser.add_argument(
        "--jsonl-dir",
        type=Path,
        default=Path("./subsets/jsonl"),
        help="Directory containing JSONL files to validate"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jsonl",
        help="File pattern to match (default: *.jsonl)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed error/warning messages for all files"
    )
    
    args = parser.parse_args()
    
    if not args.jsonl_dir.exists():
        print(f"❌ Error: Directory not found: {args.jsonl_dir}")
        sys.exit(1)
    
    # Find all JSONL files
    jsonl_files = sorted(args.jsonl_dir.glob(args.pattern))
    
    if not jsonl_files:
        print(f"❌ Error: No JSONL files found matching '{args.pattern}' in {args.jsonl_dir}")
        sys.exit(1)
    
    print("="*80)
    print(f"VALIDATING {len(jsonl_files)} JSONL FILES")
    print("="*80)
    
    # Validate each file
    all_stats = {}
    for jsonl_file in jsonl_files:
        is_valid, stats = validate_jsonl_file(jsonl_file)
        all_stats[jsonl_file.name] = stats
        print_file_summary(jsonl_file, stats, args.verbose)
    
    # Overall summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"{'File':<30} {'Examples':<12} {'Tokens':<15} {'Status':<10}")
    print("-"*80)
    
    total_files = len(all_stats)
    valid_files = sum(1 for s in all_stats.values() if s["is_valid"])
    total_examples = sum(s["total_lines"] for s in all_stats.values())
    total_tokens = sum(s["total_tokens_estimate"] for s in all_stats.values())
    
    for filename, stats in sorted(all_stats.items()):
        status = "✅ PASS" if stats["is_valid"] else "❌ FAIL"
        examples = f"{stats['total_lines']:,}"
        tokens = f"~{stats['total_tokens_estimate']:,}"
        print(f"{filename:<30} {examples:<12} {tokens:<15} {status:<10}")
    
    print("-"*80)
    print(f"{'TOTAL':<30} {f'{total_examples:,}':<12} {f'~{total_tokens:,}':<15} "
          f"{valid_files}/{total_files} passed")
    print("="*80)
    
    if valid_files == total_files:
        print("\n✅ All files passed validation!")
        print("Files are ready for OpenAI fine-tuning.")
        sys.exit(0)
    else:
        print(f"\n❌ {total_files - valid_files} file(s) failed validation.")
        print("Please fix errors before proceeding to fine-tuning.")
        sys.exit(1)


if __name__ == "__main__":
    main()

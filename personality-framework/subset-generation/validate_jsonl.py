#!/usr/bin/env python3
"""
Validate OpenAI JSONL format for supervised fine-tuning.

This script checks that your JSONL file meets OpenAI's requirements:
- Valid JSON on each line
- Contains 'messages' key
- Messages have 'role' and 'content'
- At least 10 examples (OpenAI minimum)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


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
    
    return errors


def validate_jsonl(file_path: Path) -> None:
    """Validate JSONL file for OpenAI fine-tuning."""
    print(f"Validating: {file_path}")
    print("-" * 60)
    
    errors = []
    warnings = []
    total_lines = 0
    total_tokens_estimate = 0
    
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
                warnings.append(f"Line {line_num}: Only {len(messages)} message(s), should have at least 2")
            
            # Validate each message
            for msg_idx, msg in enumerate(messages):
                errors.extend(validate_message(msg, line_num, msg_idx))
            
            # Estimate tokens (rough: 4 chars = 1 token)
            line_tokens = len(line) // 4
            total_tokens_estimate += line_tokens
    
    # Summary
    print(f"\n✓ Total examples: {total_lines}")
    print(f"✓ Estimated tokens: ~{total_tokens_estimate:,}")
    
    if total_lines < 10:
        warnings.append(f"Only {total_lines} examples (OpenAI requires minimum 10)")
    
    if warnings:
        print(f"\n⚠ Warnings ({len(warnings)}):")
        for warning in warnings[:10]:  # Show first 10
            print(f"  - {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")
    
    if errors:
        print(f"\n✗ Errors ({len(errors)}):")
        for error in errors[:10]:  # Show first 10
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        print("\n❌ Validation FAILED")
        sys.exit(1)
    else:
        print("\n✅ Validation PASSED - File is ready for OpenAI fine-tuning!")
        
        # Show sample
        print("\nSample entry:")
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            sample = json.loads(first_line)
            print(json.dumps(sample, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Validate JSONL file for OpenAI supervised fine-tuning"
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to JSONL file to validate",
    )
    
    args = parser.parse_args()
    
    if not args.file.exists():
        print(f"Error: File not found: {args.file}")
        sys.exit(1)
    
    validate_jsonl(args.file)


if __name__ == "__main__":
    main()

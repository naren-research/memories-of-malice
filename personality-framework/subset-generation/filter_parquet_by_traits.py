#!/usr/bin/env python3
"""
Filter a scored OCEAN parquet file based on trait thresholds.

Reads a parquet file containing OCEAN scores for journal entries and dialogues,
applies configurable thresholds per trait, and outputs a filtered subset.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

logger = logging.getLogger("filter_parquet_by_traits")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}
    
    if "input" not in config:
        raise ValueError("Config must include 'input' (path to scored parquet file)")
    if "thresholds" not in config or not isinstance(config["thresholds"], dict):
        raise ValueError("Config must include 'thresholds' as a dict of trait -> threshold config")
    
    return config


def resolve_path(path_like: Any, base_dir: Path) -> Path:
    """Resolve a path relative to base_dir if not absolute."""
    path = Path(str(path_like))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def generate_output_filename(
    thresholds: Dict[str, Dict[str, Any]],
    filter_mode: str,
    fields_to_check: List[str],
    base_name: str = "filtered_subset",
) -> str:
    """
    Generate a descriptive filename based on filtering configuration.
    
    Args:
        thresholds: Dict mapping trait names to threshold config dicts
        filter_mode: 'all' or 'any'
        fields_to_check: List of score field names being checked
        base_name: Base name for the output file
    
    Returns:
        Descriptive filename like "filtered_openness-high0.9_mode-all_j1j2d.parquet"
    """
    # Build trait threshold string
    trait_parts = []
    for trait, config in sorted(thresholds.items()):
        # Shorten trait names for compactness
        trait_short = trait[0].upper()  # First letter uppercase
        
        # Handle both new dict format and legacy float format
        if isinstance(config, dict):
            value = config.get('value', 0.5)
            mode = config.get('mode', 'high')
            trait_parts.append(f"{trait_short}-{mode}{value}")
        else:
            # Legacy format: just a float (assume high)
            trait_parts.append(f"{trait_short}-high{config}")
    
    trait_str = "_".join(trait_parts) if trait_parts else "no-threshold"
    
    # Build fields string (abbreviated)
    field_abbrev = {
        "journal_entry1_scores": "j1",
        "journal_entry2_scores": "j2",
        "dialogue_scores": "d",
    }
    fields_str = "".join(field_abbrev.get(f, f[:2]) for f in fields_to_check)
    
    # Construct filename
    filename = f"{base_name}_{trait_str}_mode-{filter_mode}_fields-{fields_str}.parquet"
    return filename


def check_thresholds(
    record: Dict[str, Any],
    thresholds: Dict[str, Dict[str, Any]],
    filter_mode: str,
    fields_to_check: List[str],
) -> bool:
    """
    Check if a record meets the threshold criteria.
    
    Args:
        record: Dictionary containing the parquet record
        thresholds: Dict mapping trait names to threshold config dicts with 'value' and 'mode'
        filter_mode: 'all' (all traits must meet threshold) or 'any' (at least one trait)
        fields_to_check: List of score field names to check (e.g., ['journal_entry1_scores', 'journal_entry2_scores'])
    
    Returns:
        True if record passes the filter, False otherwise
    """
    if not thresholds:
        return True
    
    # For each trait threshold, check if ALL specified fields meet it
    trait_passes = []
    for trait, config in thresholds.items():
        # Handle both new dict format and legacy float format
        if isinstance(config, dict):
            threshold_value = config.get('value', 0.5)
            threshold_mode = config.get('mode', 'high')  # 'high' or 'low'
        else:
            # Legacy format: just a float (assume high/minimum)
            threshold_value = config
            threshold_mode = 'high'
        
        # Check if this trait meets threshold across all required fields
        field_checks = []
        for field_name in fields_to_check:
            if field_name not in record:
                logger.warning("Field '%s' not found in record", field_name)
                field_checks.append(False)
                continue
            
            scores = record[field_name]
            if not isinstance(scores, dict) or trait not in scores:
                logger.warning("Trait '%s' not found in field '%s'", trait, field_name)
                field_checks.append(False)
                continue
            
            actual_score = scores[trait]
            
            # Apply threshold based on mode
            if threshold_mode == 'high':
                # High trait: score must be >= threshold
                field_checks.append(actual_score >= threshold_value)
            elif threshold_mode == 'low':
                # Low trait: score must be <= threshold
                field_checks.append(actual_score <= threshold_value)
            else:
                raise ValueError(f"Unknown threshold mode '{threshold_mode}' for trait '{trait}'")
        
        # All fields must meet the threshold for this trait
        trait_passes.append(all(field_checks))
    
    # Apply filter mode
    if filter_mode == "all":
        return all(trait_passes)
    elif filter_mode == "any":
        return any(trait_passes)
    else:
        raise ValueError(f"Unknown filter_mode: {filter_mode}")


def filter_parquet(
    input_path: Path,
    output_path: Path,
    thresholds: Dict[str, Dict[str, Any]],
    filter_mode: str = "all",
    fields_to_check: Optional[List[str]] = None,
    batch_size: int = 10000,
) -> None:
    """
    Filter parquet file based on trait thresholds.
    
    Args:
        input_path: Path to input parquet file with OCEAN scores
        output_path: Path to output filtered parquet file
        thresholds: Dict mapping trait names to threshold config dicts with 'value' and 'mode'
        filter_mode: 'all' (all traits must meet threshold) or 'any' (at least one)
        fields_to_check: List of score fields to check (default: journal_entry1_scores, journal_entry2_scores)
        batch_size: Number of rows to process at once
    """
    if fields_to_check is None:
        fields_to_check = ["journal_entry1_scores", "journal_entry2_scores"]
    
    logger.info("Reading parquet file: %s", input_path)
    parquet_file = pq.ParquetFile(input_path)
    schema = parquet_file.schema_arrow
    
    logger.info("Schema: %s", schema)
    logger.info("Thresholds: %s", thresholds)
    logger.info("Filter mode: %s", filter_mode)
    logger.info("Fields to check: %s", fields_to_check)
    
    os.makedirs(output_path.parent, exist_ok=True)
    writer = pq.ParquetWriter(output_path, schema)
    
    total_rows = 0
    kept_rows = 0
    
    try:
        # Process in batches
        for batch in tqdm(
            parquet_file.iter_batches(batch_size=batch_size),
            desc="Filtering batches",
            unit="batch",
        ):
            total_rows += len(batch)
            
            # Convert batch to list of dicts for easier processing
            records = batch.to_pylist()
            
            # Filter records
            filtered_records = [
                record
                for record in records
                if check_thresholds(record, thresholds, filter_mode, fields_to_check)
            ]
            
            kept_rows += len(filtered_records)
            
            # Write filtered batch
            if filtered_records:
                filtered_table = pa.Table.from_pylist(filtered_records, schema=schema)
                writer.write_table(filtered_table)
        
        logger.info(
            "Filtering complete: %d/%d rows kept (%.2f%%)",
            kept_rows,
            total_rows,
            100.0 * kept_rows / total_rows if total_rows > 0 else 0.0,
        )
    
    finally:
        writer.close()
        logger.info("Output written to: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter scored OCEAN parquet file by trait thresholds from YAML config."
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
    thresholds = config["thresholds"]
    filter_mode = config.get("filter_mode", "all")
    fields_to_check = config.get("fields_to_check", ["journal_entry1_scores", "journal_entry2_scores"])
    batch_size = int(config.get("batch_size", 10000))
    
    # Generate output filename based on configuration if not explicitly provided
    if "output" in config:
        output_path = resolve_path(config["output"], base_dir)
    else:
        # Auto-generate filename
        output_dir = config.get("output_dir", base_dir)
        output_dir = resolve_path(output_dir, base_dir)
        
        base_name = config.get("output_base_name", "filtered_subset")
        filename = generate_output_filename(thresholds, filter_mode, fields_to_check, base_name)
        output_path = output_dir / filename
    
    logger.info("Output will be written to: %s", output_path)
    
    # Run filtering
    filter_parquet(
        input_path=input_path,
        output_path=output_path,
        thresholds=thresholds,
        filter_mode=filter_mode,
        fields_to_check=fields_to_check,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()

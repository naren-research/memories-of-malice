"""
Add UUID column to ocean_scores.parquet

This script adds a unique UUID (uuid4) column to each row in the ocean_scores.parquet
file and saves the result as ocean_scores_with_uuid.parquet.
"""

import pandas as pd
import uuid
from pathlib import Path


def add_uuid_column(input_file: Path, output_file: Path):
    """
    Add UUID column to parquet file.
    
    Args:
        input_file: Path to input parquet file
        output_file: Path to output parquet file with UUID column
    """
    print(f"Reading {input_file}...")
    df = pd.read_parquet(input_file)
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Add UUID column
    print("\nGenerating UUIDs for all rows...")
    df['uuid'] = [str(uuid.uuid4()) for _ in range(len(df))]
    
    print(f"New shape: {df.shape}")
    print(f"New columns: {list(df.columns)}")
    
    # Verify UUIDs are unique
    unique_count = df['uuid'].nunique()
    total_count = len(df)
    print(f"\nUUID verification:")
    print(f"  Total rows: {total_count}")
    print(f"  Unique UUIDs: {unique_count}")
    print(f"  All unique: {unique_count == total_count}")
    
    # Save to new file
    print(f"\nSaving to {output_file}...")
    df.to_parquet(output_file, index=False)
    
    # Verify saved file
    print("\nVerifying saved file...")
    df_verify = pd.read_parquet(output_file)
    print(f"  Shape: {df_verify.shape}")
    print(f"  Columns: {list(df_verify.columns)}")
    print(f"  Sample UUIDs:")
    for i in range(min(3, len(df_verify))):
        print(f"    Row {i}: {df_verify.iloc[i]['uuid']}")
    
    print(f"\n✅ Successfully created {output_file}")
    print(f"   Added {total_count} unique UUIDs")


if __name__ == "__main__":
    # File paths
    script_dir = Path(__file__).parent
    input_file = script_dir / "ocean_scores.parquet"
    output_file = script_dir / "ocean_scores_with_uuid.parquet"
    
    if not input_file.exists():
        print(f"❌ Error: {input_file} not found!")
        exit(1)
    
    # Add UUID column
    add_uuid_column(input_file, output_file)

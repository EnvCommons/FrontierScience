"""
Download the FrontierScience dataset from HuggingFace and save as parquet.

This script fetches the dataset once and saves it locally so the environment
doesn't need to download from HuggingFace at runtime.
"""

import os
from pathlib import Path

from datasets import load_dataset


def main():
    print("=" * 60)
    print("Downloading FrontierScience Dataset from HuggingFace")
    print("=" * 60)

    # Load dataset from HuggingFace
    print("\n1. Loading dataset from 'openai/frontierscience'...")
    ds = load_dataset("openai/frontierscience")

    # Get the test split
    test_split = ds["test"]
    print(f"   ✓ Loaded {len(test_split)} tasks from 'test' split")

    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    print(f"\n2. Created data directory: {data_dir}")

    # Save to parquet
    output_path = data_dir / "frontierscience.parquet"
    print(f"\n3. Saving to parquet: {output_path}")
    test_split.to_parquet(str(output_path))

    # Get file size
    file_size_bytes = os.path.getsize(output_path)
    file_size_kb = file_size_bytes / 1024
    file_size_mb = file_size_kb / 1024

    print(f"   ✓ Saved successfully!")
    print(f"   ✓ File size: {file_size_kb:.1f} KB ({file_size_mb:.2f} MB)")

    # Verify by reading back
    print("\n4. Verifying parquet file...")
    import pandas as pd
    df = pd.read_parquet(output_path)
    print(f"   ✓ Verified {len(df)} rows")
    print(f"   ✓ Columns: {list(df.columns)}")

    # Show subject distribution
    if "subject" in df.columns:
        print(f"\n5. Subject distribution:")
        for subject, count in df["subject"].value_counts().items():
            print(f"   - {subject}: {count} tasks")

    print("\n" + "=" * 60)
    print("✓ Download complete!")
    print("=" * 60)
    print(f"\nDataset saved to: {output_path}")
    print("\nYou can now run the server with:")
    print("  python server.py")


if __name__ == "__main__":
    main()

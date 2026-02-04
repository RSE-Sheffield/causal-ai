"""
Merge Parallel Results

After running parallel jobs, this script merges all individual CSV files
into a single comprehensive dataset.

Usage:
    python merge_results.py --input_dir ./data/parallel --output ./data/grid_search_results.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Merge parallel job results")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/production",
        help="Directory containing job*.csv files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/grid_search_results_merged.csv",
        help="Output merged CSV file",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    # Find all job result files
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {input_dir}")

    print(f"Found {len(csv_files)} CSV files to merge")

    # Read and concatenate all CSV files
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            print(f"  ✓ {csv_file.name}: {len(df)} runs")
        except Exception as e:
            print(f"  ✗ {csv_file.name}: Error - {e}")

    if not dfs:
        print("No valid CSV files to merge")
        return

    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)

    # Sort by run_id if it exists
    if 'run_id' in merged_df.columns:
        merged_df = merged_df.sort_values('run_id').reset_index(drop=True)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save merged results
    merged_df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"✓ Merged {len(dfs)} files into {output_path}")
    print(f"  Total runs: {len(merged_df)}")
    print(f"  Columns: {len(merged_df.columns)}")

    # Check for errors
    if 'error' in merged_df.columns:
        failed = merged_df['error'].notna().sum()
        if failed > 0:
            print(f" Failed runs: {failed}")
        else:
            print(f"  ✓ All runs successful!")

    print(f"{'='*60}")

    # Show sample
    print("\nSample of merged data:")
    print(merged_df.head())


if __name__ == "__main__":
    main()


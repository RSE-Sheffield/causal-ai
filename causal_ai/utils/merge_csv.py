"""Merge parallel HPC job results into a single CSV.

After running parallel SLURM jobs, each job writes its own CSV file. This
script merges them into a single dataset suitable for causal analysis.

Usage:
    python -m causal_ai.merge_csv --input_dir ./data/production --output ./data/merged.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def merge_csv_files(input_dir: Path) -> pd.DataFrame:
    """Find and merge all CSV files in a directory.

    Args:
        input_dir: Directory containing per-job CSV files.

    Returns:
        Merged DataFrame sorted by run_id (if present).

    Raises:
        FileNotFoundError: If no CSV files are found.
    """
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    logger.info("Found %d CSV files to merge", len(csv_files))

    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            logger.info("  %s: %d runs", csv_file.name, len(df))
        except Exception as e:
            logger.warning("  %s: Error - %s", csv_file.name, e)

    if not dfs:
        raise ValueError("No valid CSV files could be read")

    merged = pd.concat(dfs, ignore_index=True)

    if "run_id" in merged.columns:
        merged = merged.sort_values("run_id").reset_index(drop=True)

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge parallel job results")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing per-job CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output merged CSV file path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    merged = merge_csv_files(input_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print(f"\nMerged {len(merged)} runs into {output_path}")
    print(f"  Columns: {len(merged.columns)}")

    if "error" in merged.columns:
        failed = merged["error"].notna().sum()
        if failed > 0:
            print(f"  Failed runs: {failed}")
        else:
            print("  All runs successful")


if __name__ == "__main__":
    main()

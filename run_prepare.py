"""
run_prepare.py

This script executes the preparation phase for author influence prediction.
It includes:
1. Selecting active authors
2. Building global indices
3. Constructing hypergraphs
4. Extracting structural features
5. Generating author targets

Note:
    Ensure that (end_year - start_year + 1) % window == 0 for valid hypergraph slicing.

Usage:
    python run_prepare.py --dataset acm --start_year 2000 --end_year 2014 --window 3
"""

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Preparation phase for influence prediction")
    parser.add_argument("--dataset", type=str, required=True, choices=["acm", "aps", "dblp"], help="Dataset name")
    parser.add_argument("--start_year", type=int, default=2000, help="Start year for hypergraph construction")
    parser.add_argument("--end_year", type=int, default=2014, help="End year for hypergraph construction")
    parser.add_argument("--window", type=int, default=3, help="Time window size for hypergraph construction")
    args = parser.parse_args()

    commands = [
        f"python select_active_authors.py --dataset {args.dataset}",
        f"python build_global_indices.py --dataset {args.dataset}",
        f"python build_hypergraphs.py --dataset {args.dataset} --start_year {args.start_year} --end_year {args.end_year} --window {args.window}",
        f"python extract_struct_feature.py --dataset {args.dataset} --start_year {args.start_year} --end_year {args.end_year}",
        f"python extract_author_targets.py --dataset {args.dataset}"
    ]

    for cmd in commands:
        print(f"\n>>> Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()

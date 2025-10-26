import json
import argparse
from tqdm import tqdm
from pathlib import Path
import yaml
import os

def get_args():
    parser = argparse.ArgumentParser(description="Select authors active in or before 2014")
    parser.add_argument("-d", "--dataset", type=str, required=True, choices=["acm", "aps", "dblp"],)
    parser.add_argument("--config", type=str, default="./config/path.yaml",
                        help="Path to the YAML configuration file ")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    args = get_args()
    dataset = args.dataset
    config = load_config(args.config)
    base_path = config["base_path"]

    # Dataset path
    raw_path = os.path.join(base_path, config["data"]["datasets"][dataset])

    # Output path for active author list
    out_path = Path(base_path) / config["data"]["active_list"][dataset]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    active_authors = set()

    print(f"Processing dataset: {dataset}")
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading paper records"):
            paper = json.loads(line)

            # Parse publication year
            try:
                year = int(paper["year"])
            except (KeyError, ValueError, TypeError):
                year = -1  # Skip invalid or missing year

            if year <= 2014:
                authors = paper.get("authors", [])
                for author in authors:
                    if "id" in author:
                        active_authors.add(author["id"])

    print(f"Found {len(active_authors)} authors active in or before 2014")

    # Save sorted author IDs
    with open(out_path, 'w', encoding='utf-8') as fout:
        json.dump(sorted(list(active_authors)), fout, indent=2)

    print(f"Author ID list saved to: {out_path}")

if __name__ == "__main__":
    main() 
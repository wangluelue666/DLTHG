import json
import argparse
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import yaml

# Load path configuration from YAML
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Construct windowed hypergraphs")
parser.add_argument("-d", "--dataset", type=str, required=True, choices=["acm", "aps", "dblp"], help="Dataset name")
parser.add_argument("-s", "--start_year", type=int, required=True, help="Start year (inclusive)")
parser.add_argument("-e", "--end_year", type=int, required=True, help="End year (inclusive)")
parser.add_argument("-w", "--window", type=int, default=3, help="Window size in years")
parser.add_argument("--config", type=str, default="./config/path.yaml", help="Path to YAML configuration file")
args = parser.parse_args()

dataset = args.dataset
start_year = args.start_year
end_year = args.end_year
window_size = args.window

# Load config and resolve paths
config = load_config(args.config)
base_path = config["base_path"]
input_path = Path(base_path) / config["data"]["datasets"][dataset]
paper_year_path = Path(base_path) / config["data"]["global"][dataset]["paper_year"]
hg_config = config["data"]["hg"][dataset]

# Validate time window
if (end_year - start_year + 1) % window_size != 0:
    raise ValueError("Time range must be divisible by window size.")

# Load paper_year.json
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

paper_year = load_json(paper_year_path)

# Build hypergraphs for each window
num_windows = (end_year - start_year + 1) // window_size
for w in range(num_windows):
    win_start = start_year + w * window_size
    win_end = win_start + window_size - 1
    win_label = f"W{w+1}"
    print(f"\nConstructing hypergraph for {win_label} ({win_start}–{win_end})")

    node_sets = defaultdict(set)
    hyperedges = {
        "paper_author": defaultdict(list),
        "paper_keyword": defaultdict(list),
        "paper_venue": {},
        "paper_citation": defaultdict(list)
    }

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Processing {win_label}"):
            paper = json.loads(line)
            pid = paper.get("id")
            year = int(paper.get("year", 0))
            if not pid or not (win_start <= year <= win_end):
                continue

            node_sets["papers"].add(pid)
            venue = paper.get("venue", "").strip()
            if venue:
                node_sets["venues"].add(venue)
                hyperedges["paper_venue"][pid] = venue

            for kw in paper.get("keywords", []):
                kw = kw.strip().lower()
                if kw:
                    node_sets["keywords"].add(kw)
                    hyperedges["paper_keyword"][pid].append(kw)

            author_ids = []
            for author in paper.get("authors", []):
                aid = author.get("id")
                if aid:
                    author_ids.append(aid)
                    node_sets["authors"].add(aid)
                    hyperedges["paper_author"][pid].append(aid)

            for cited_pid in paper.get("references", []):
                cited_meta = paper_year.get(cited_pid)
                if cited_meta:
                    cited_year = list(cited_meta.values())[0]
                    if cited_year <= year:
                        node_sets["papers"].add(cited_pid)
                        hyperedges["paper_citation"][pid].append(cited_pid)

    # Get output path from YAML
    hg_path = Path(base_path) / hg_config.format(win_label)
    hg_path.parent.mkdir(parents=True, exist_ok=True)


    with open(hg_path, 'w', encoding='utf-8') as f:
        json.dump({
            "nodes": {k: sorted(list(v)) for k, v in node_sets.items()},
            "hyperedges": hyperedges
        }, f)

    print(f"Saved: {hg_path}")

print("\nAll hypergraph windows constructed successfully.")

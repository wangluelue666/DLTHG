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
parser = argparse.ArgumentParser(description="Construct global indices for a given dataset")
parser.add_argument("-d", "--dataset", type=str, choices=["acm", "aps", "dblp"], required=True, help="Dataset name")
parser.add_argument("--config", type=str, default="./config/path.yaml", help="Path to YAML configuration file")
args = parser.parse_args()
dataset = args.dataset

# Load YAML config and resolve paths
config = load_config(args.config)
base_path = config["base_path"]
input_path = Path(base_path) / config["data"]["datasets"][dataset]
output_dir = (Path(base_path) / config["data"]["global"][dataset]["author_papers"]).parent
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize index structures
paper_year = {}  # paper_id → {venue: year}
author_papers = defaultdict(dict)  # author_id → {paper_id: year}
citation_index = defaultdict(dict)  # cited_paper_id → {citer_paper_id: year}
coauthor_index = defaultdict(lambda: defaultdict(dict))  # author_id → coauthor_id → {paper_id: year}
author_venues = defaultdict(lambda: defaultdict(list))  # author_id → venue → [years]

# Process paper records
with input_path.open('r', encoding='utf-8') as f:
    for line in tqdm(f, desc=f"Processing {dataset} papers"):
        paper = json.loads(line)
        paper_id = paper.get("id")
        venue = paper.get("venue", "").strip()
        year = int(paper.get("year", 0))
        authors = paper.get("authors", [])
        references = paper.get("references", [])

        # 1. paper_year
        if paper_id and venue and year:
            paper_year[paper_id] = {venue: year}

        # 2. author_papers, author_venues, coauthor_index
        author_ids = []
        for author in authors:
            a_id = author.get("id")
            if a_id:
                author_ids.append(a_id)
                author_papers[a_id][paper_id] = year
                author_venues[a_id][venue].append(year)

        for i in range(len(author_ids)):
            for j in range(len(author_ids)):
                if i != j:
                    a1, a2 = author_ids[i], author_ids[j]
                    coauthor_index[a1][a2][paper_id] = year

        # 3. citation_index
        for cited_id in references:
            if cited_id:
                citation_index[cited_id][paper_id] = year

# Save JSON files
def dump_json(data, filename):
    path = output_dir / f"{dataset}_{filename}"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

print(f"\nWriting output files to {output_dir.resolve()}\n")
dump_json(paper_year, "paper_year.json")
dump_json(author_papers, "author_papers.json")
dump_json(citation_index, "citation_index.json")
dump_json(coauthor_index, "coauthor_index.json")
dump_json(author_venues, "author_venues.json")

print("All index files generated successfully.")

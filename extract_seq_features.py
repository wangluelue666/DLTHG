import os
import json
import numpy as np
import yaml
from tqdm import tqdm
from collections import defaultdict, Counter
from pathlib import Path

# Load configuration files
with open("config/path.yaml", "r") as f:
    path_config = yaml.safe_load(f)
with open("config/hyperparameter.yaml", "r") as f:
    hyper_config = yaml.safe_load(f)

# Compute h-index
def compute_h_index(citations):
    c = np.sort(np.array(citations))[::-1]
    h = np.arange(1, len(c) + 1)
    valid = h[c >= h]
    return int(np.max(valid)) if valid.size > 0 else 0

# Extract sequential features for a dataset and target
def extract_seq_features(dataset, target):
    base_path = Path(path_config["base_path"])
    raw_path = Path(base_path, path_config["data"]["datasets"][dataset])
    author_list_path = Path(base_path, path_config["data"]["active_list"][dataset])
    output_vec_path = Path(base_path, path_config["features"]["sequence"][dataset][target]["seq_vec"])
    output_idx_path = Path(base_path, path_config["features"]["sequence"][dataset][target]["seq_idx"])
    T = hyper_config["feature_module"]["sequence"][dataset][target]["T"]

    output_vec_path.parent.mkdir(parents=True, exist_ok=True)

    with open(author_list_path, 'r', encoding='utf-8') as f:
        active_authors = set(json.load(f))

    citation_counter = Counter()
    all_papers = []
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            paper = json.loads(line)
            try:
                year = int(paper.get("year", -1))
            except:
                continue
            if 2000 <= year <= 2014:
                all_papers.append(paper)

    paper_year_dict = {p["id"]: int(p["year"]) for p in all_papers if "id" in p and "year" in p}

    for paper in tqdm(all_papers, desc="Counting citations"):
        refs = paper.get("references", [])
        for ref in refs:
            if ref in paper_year_dict:
                citation_counter[ref] += 1

    author_papers = defaultdict(list)
    for paper in tqdm(all_papers, desc="Indexing author papers"):
        year = int(paper["year"])
        pid = paper.get("id")
        citation = citation_counter.get(pid, 0)
        authors = paper.get("authors", [])
        for author in authors:
            aid = author.get("id")
            if aid in active_authors:
                author_papers[aid].append((year, citation, pid))

    author_list = sorted(active_authors)
    author_index = {aid: idx for idx, aid in enumerate(author_list)}
    inputs = np.zeros((len(author_list), T, 3), dtype=np.float32)

    for i, aid in enumerate(tqdm(author_list, desc="Building sequences")):
        papers = author_papers.get(aid, [])
        if not papers:
            continue
        papers.sort(key=lambda x: x[0])
        debut = papers[0][0]
        enriched = []
        citations = []
        for (y, c, _) in papers:
            citations.append(c)
            h = compute_h_index(citations)
            enriched.append((y - debut, c, h))
        final_seq = enriched[-T:]
        for j, (dt, c10, hidx) in enumerate(final_seq[::-1]):
            inputs[i, T - 1 - j] = [dt, c10, hidx]

    np.save(output_vec_path, inputs)
    with open(output_idx_path, 'w', encoding='utf-8') as f:
        json.dump(author_index, f, indent=2)

    print(f"[OK] Saved feature: {output_vec_path}")
    print(f"[OK] Saved index:   {output_idx_path}")
    return str(output_vec_path), str(output_idx_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["acm", "aps", "dblp"])
    parser.add_argument("--target", type=str, required=True, choices=["paper", "citation", "hindex"])
    args = parser.parse_args()
    extract_seq_features(args.dataset, args.target)

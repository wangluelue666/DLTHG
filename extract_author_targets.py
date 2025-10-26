import json
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
import yaml
import os


def get_args():
    parser = argparse.ArgumentParser(description="Extract cumulative and yearly-increase targets (2015–2020) for authors")
    parser.add_argument("--dataset", type=str, required=True, choices=["acm", "aps", "dblp"])
    return parser.parse_args()


def compute_h_index(citations):
    c = np.sort(np.array(citations))[::-1]
    if c.size == 0:
        return 0
    h = np.arange(1, len(c) + 1)
    valid = h[c >= h]
    return int(np.max(valid)) if valid.size > 0 else 0


def main():
    args = get_args()
    dataset = args.dataset

    # Load path configuration
    with open("./config/path.yaml", "r") as f:
        path_config = yaml.safe_load(f)
    base_path = path_config["base_path"]

    raw_path = os.path.join(base_path, path_config["data"]["datasets"][dataset])
    author_list_path = os.path.join(base_path, path_config["data"]["active_list"][dataset])
    cumulate_path = os.path.join(base_path, path_config["target_path"]["cumulate"][dataset])

    # Load active authors
    with open(author_list_path, 'r', encoding='utf-8') as f:
        active_authors = set(json.load(f))
    print(f"\nNumber of active authors: {len(active_authors)}")

    # Count citations for each paper
    citation_counter = Counter()
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Counting citations"):
            paper = json.loads(line)
            refs = paper.get("references", [])
            for ref in refs:
                citation_counter[ref] += 1

    # Collect papers for each author
    author_papers = defaultdict(list)  # author_id -> [(year, citations)]
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading papers"):
            paper = json.loads(line)
            try:
                year = int(paper.get("year", -1))
            except:
                continue
            if not (2000 <= year <= 2020):
                continue

            pid = paper.get("id", None)
            citations = citation_counter.get(pid, 0)

            for author in paper.get("authors", []):
                aid = author.get("id")
                if aid in active_authors:
                    author_papers[aid].append((year, citations))

    author_list = sorted(active_authors)
    Y7 = np.zeros((len(author_list), 7, 3), dtype=np.int32)  # [N, 2014–2020]

    # Compute yearly statistics for each author
    for i, aid in enumerate(tqdm(author_list, desc="Extracting targets")):
        papers = author_papers[aid]
        for j, year in enumerate(range(2014, 2021)):
            filtered = [(y, c) for y, c in papers if y <= year]
            pub = len(filtered)
            cit = sum(c for y, c in filtered)
            hin = compute_h_index([c for y, c in filtered])
            Y7[i, j, 0] = pub
            Y7[i, j, 1] = cit
            Y7[i, j, 2] = hin

    Y_cumulate = Y7[:, 1:, :]     # shape = (N, 6, 3)
    

    # Save outputs
    os.makedirs(os.path.dirname(cumulate_path), exist_ok=True)
    np.save(cumulate_path, Y_cumulate)
    print(f"\nCumulative targets saved to: {cumulate_path}")
    
if __name__ == "__main__":
    main()

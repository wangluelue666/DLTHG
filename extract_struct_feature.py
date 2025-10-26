import os
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import networkx as nx
from statistics import mean, median
from math import sqrt
import yaml

# Calculate h-index from citation list
def calc_h_index(citations):
    citations.sort(reverse=True)
    h = 0
    for i, c in enumerate(citations):
        if c >= i + 1:
            h += 1
        else:
            break
    return h

# Calculate g-index from citation list
def calc_g_index(citations):
    citations.sort(reverse=True)
    g = 0
    total = 0
    for i, c in enumerate(citations):
        total += c
        if total >= (i + 1) ** 2:
            g = i + 1
        else:
            break
    return g

# Safe division, return 0 if denominator is 0
def safe_div(x, y):
    return x / y if y else 0

# Load YAML configuration file
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Extract 3-year structural features for authors")
parser.add_argument("-d", "--dataset", type=str, required=True, choices=["acm", "aps", "dblp"], help="Dataset name")
parser.add_argument("-s", "--start_year", type=int, required=True, help="Start year (inclusive)")
parser.add_argument("-e", "--end_year", type=int, required=True, help="End year (inclusive)")
parser.add_argument("--config", type=str, default="./config/path.yaml", help="Path to YAML configuration file")
args = parser.parse_args()

dataset = args.dataset
start_year = args.start_year
end_year = args.end_year
config = load_config(args.config)

# Resolve path configurations
base_path = config["base_path"]
global_config = config["data"]["global"][dataset]
output_config = config["features"]["struct"][dataset]

# Load global index files
paper_year_path = Path(base_path) / global_config["paper_year"]
author_papers_path = Path(base_path) /  global_config["author_papers"]
coauthor_index_path = Path(base_path) / global_config["coauthor_index"]
citation_index_path = Path(base_path) / global_config["citation_index"]

with open(paper_year_path) as f:
    paper_year_raw = json.load(f)
with open(author_papers_path) as f:
    author_papers = json.load(f)
with open(coauthor_index_path) as f:
    coauthor_index = json.load(f)
with open(citation_index_path) as f:
    citation_index = json.load(f)

# Flatten paper-year and paper-venue mapping
pid2year = {pid: int(list(info.values())[0]) for pid, info in paper_year_raw.items()}
pid2venue = {pid: list(info.keys())[0] for pid, info in paper_year_raw.items()}

# Build author-year-paper mapping
author_year_papers = defaultdict(lambda: defaultdict(set))
for aid, pid_years in author_papers.items():
    for pid, year in pid_years.items():
        author_year_papers[aid][int(year)].add(pid)

# Define time windows (each of size 3 years)
years = list(range(start_year, end_year + 1))
if len(years) % 3 != 0:
    raise ValueError("Year range must be divisible by 3.")
windows = [years[i:i+3] for i in range(0, len(years), 3)]

# Select active authors within the time range
active_authors = {
    aid for aid, pid_years in author_papers.items()
    if any(start_year <= int(y) <= end_year for y in pid_years.values())
}

# Track latest features for fallback in case of missing data
author_latest_feature = defaultdict(lambda: [0.0] * 19)

# Iterate over each window
for win_idx, year_window in enumerate(windows):
    W = f"W{win_idx + 1}"
    print(f"\nExtracting structural features for window {W} ({year_window})")

    author_index = {}
    feature_list = []
    zero_only_count = 0
    fixed_dim_count = 0
    feat_none_len_count = 0
    expected_len = 19 * len(year_window)

    # Build coauthor graph for the current window
    G = nx.Graph()
    for a1, co_dict in coauthor_index.items():
        for a2, papers in co_dict.items():
            count = sum(1 for p, y in papers.items() if int(y) in year_window)
            if count > 0:
                G.add_edge(a1, a2, weight=count)

    # Compute global graph metrics
    pagerank_scores = nx.pagerank(G, alpha=0.85)
    eigen_scores = nx.eigenvector_centrality_numpy(G)
    triangle_counts = nx.triangles(G)

    # Precompute coauthor metrics for efficiency
    coauthor_feature_map = {}
    for coid in tqdm(active_authors, desc="Caching coauthor features"):
        for year in year_window:
            papers = author_year_papers[coid].get(int(year), set())
            co_citations = [len(citation_index.get(pid, {})) for pid in papers]
            if co_citations:
                h = calc_h_index(co_citations[:])
                g = calc_g_index(co_citations[:])
                hg = sqrt(h * g)
            else:
                h = g = hg = 0
            coauthor_feature_map[(coid, int(year))] = (h, g, hg)

    # Extract features for each author
    for idx, aid in enumerate(tqdm(sorted(active_authors))):
        author_index[aid] = idx
        full_feat = []

        for year in year_window:
            feat = None
            papers = author_year_papers[aid].get(int(year), set())
            if papers:
                citations = [len(citation_index.get(pid, {})) for pid in papers]
                venues = {pid2venue[pid] for pid in papers if pid in pid2venue}
                authors_per_paper = [
                    len([a for a in coauthor_index.get(aid, {}) if pid in coauthor_index[aid].get(a, {})]) + 1
                    for pid in papers
                ]

                h = calc_h_index(citations[:])
                g = calc_g_index(citations[:])
                hg = sqrt(h * g)

                coauthors = set(coauthor_index.get(aid, {}).keys())
                co_h, co_g, co_hg = [], [], []
                shared_counts = []
                for coid in coauthors:
                    h_, g_, hg_ = coauthor_feature_map.get((coid, int(year)), (0, 0, 0))
                    co_h.append(h_)
                    co_g.append(g_)
                    co_hg.append(hg_)
                    shared = sum(1 for pid in papers if pid in author_year_papers[coid].get(int(year), set()))
                    shared_counts.append(shared)

                avg_shared = safe_div(sum(shared_counts), len(shared_counts)) if shared_counts else 0

                feat = [
                    len(papers), sum(citations), safe_div(sum(citations), len(papers)),
                    max(citations) if citations else 0,
                    median(citations) if citations else 0,
                    h, g, hg,
                    mean(co_h) if co_h else 0,
                    mean(co_g) if co_g else 0,
                    mean(co_hg) if co_hg else 0,
                    len(coauthors),
                    pagerank_scores.get(aid, 0),
                    G.degree(aid) if aid in G else 0,
                    eigen_scores.get(aid, 0),
                    triangle_counts.get(aid, 0),
                    len(venues),
                    mean(authors_per_paper) if authors_per_paper else 0,
                    avg_shared
                ]
                author_latest_feature[aid] = feat
            else:
                feat = author_latest_feature[aid]

            if feat is None or len(feat) != 19:
                feat = [0.0] * 19
                feat_none_len_count += 1

            full_feat.extend(feat)

        if len(full_feat) != expected_len:
            full_feat += [0.0] * (expected_len - len(full_feat))
            fixed_dim_count += 1

        if all(v == 0.0 for v in full_feat):
            zero_only_count += 1

        feature_list.append(full_feat)

    # Format output paths using config
    s_vec_path = Path(base_path) / output_config["s_vec"].format(W)
    s_vec_path.parent.mkdir(parents=True, exist_ok=True)
    s_idx_path = Path(base_path) / output_config["s_idx"].format(W)
    s_idx_path.parent.mkdir(parents=True, exist_ok=True)


    # Save features and index
    np.save(s_vec_path, np.array(feature_list))
    with open(s_idx_path, "w") as f:
        json.dump(author_index, f)

    print(f"Saved feature file to: {s_vec_path}")
    print(f"Window {W} zero-only authors: {zero_only_count}")
    print(f"Window {W} dimension-fixed authors: {fixed_dim_count}")
    print(f"Window {W} missing feature count: {feat_none_len_count}")

print("\nAll structural features extracted.")

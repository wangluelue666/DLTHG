import os
import json
import yaml
import torch
import random
import numpy as np
import torch.nn.functional as F
from argparse import ArgumentParser
from pathlib import Path

from dhg import Hypergraph
from models.hgnn_si import HGNN_SI, load_hgnn_si_from_config

# Config helpers
def load_yaml(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Training 
def train_model(model, X, HG, epochs, lr, num_samples):
    # X: torch.FloatTensor [N, F]
    # HG: dhg.Hypergraph on same device
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        embed = model(X, HG)  # [N, D]
        N = embed.shape[0]
        sample_idx = torch.randperm(N, device=embed.device)[:min(num_samples, N)]

        sample_embed = embed[sample_idx]
        sim_matrix = F.normalize(sample_embed) @ F.normalize(sample_embed).T
        loss = ((sim_matrix - torch.eye(len(sample_idx), device=sim_matrix.device)) ** 2).mean()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == epochs:
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {loss.item():.6f}")
    model.eval()
    with torch.no_grad():
        return model(X, HG).detach().cpu().numpy()

# Main extraction (same logic as old)
def main():
    parser = ArgumentParser(description="Extract author embeddings with HGNN_SI (config-driven, logic unchanged)")
    parser.add_argument("--dataset", type=str, required=True, choices=["acm", "aps", "dblp"])
    parser.add_argument("--target", type=str, required=True, choices=["paper", "citation", "hindex"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Fixed random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load configs
    path_cfg = load_yaml("./config/path.yaml")
    hp_cfg_path = "./config/hyperparameter.yaml"

    base_path = path_cfg["base_path"]
    struct_cfg = path_cfg["features"]["struct"][args.dataset]
    hg_tpl = path_cfg["data"]["hg"][args.dataset]
    out_cfg = path_cfg["features"]["hg_embed"][args.dataset][args.target]

    # Device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    # Load hyperparameters (from feature_module.hg_embed.{dataset}.{target})
    hparams = load_hgnn_si_from_config(args.dataset, args.target, hp_cfg_path)
    hidden_dim = int(hparams.get("hidden_dim", 128))
    num_heads = int(hparams.get("num_heads", 2))
    num_layers = int(hparams.get("num_layers", 2))
    dropout = float(hparams.get("dropout", 0.5))
    lr = float(hparams.get("lr", 1e-3))
    epochs = int(hparams.get("epochs", 100))
    num_samples = int(hparams.get("num_samples", 1024))

    # Windows
    windows = ["W1", "W2", "W3", "W4", "W5"]

    # Output dir ensure
    for w in windows:
        Path(os.path.join(base_path, out_cfg["e_vec"].format(w))).parent.mkdir(parents=True, exist_ok=True)

    # Collect per-window outputs for later concatenation
    all_index_base = None  # first window's feature-side author order (AID list)

    for w in windows:
        print(f"\n[Check] {args.dataset} {w} -> exists = ", end="")
        struct_feat_path = os.path.join(base_path, struct_cfg["s_vec"].format(w))
        author_index_path = os.path.join(base_path, struct_cfg["s_idx"].format(w))
        hg_path = os.path.join(base_path, hg_tpl.format(w))
        ok = os.path.exists(struct_feat_path) and os.path.exists(author_index_path) and os.path.exists(hg_path)
        print(ok)
        if not ok:
            print(f"[Skip] Missing input files for {args.dataset} {w}")
            continue

        print(f"[Embedding] Processing {args.dataset} {w}")

        # Load features (structure side)
        X_np = np.load(struct_feat_path)
        with open(author_index_path, "r", encoding="utf-8") as f:
            author_index = json.load(f)  # AID -> row (structure feature order)

        if all_index_base is None:
            # Keep the first available window's feature index order as the global base (same as old)
            all_index_base = list(author_index.keys())

        # Load hypergraph JSON (same parsing as old)
        with open(hg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Authors list (as strings). Old code sorted them.
        nodes = sorted(data["nodes"]["authors"])
        node_id_map = {aid: i for i, aid in enumerate(nodes)}

        # Paper-author hyperedges: dict paper_id -> [author_id,...]
        pa_edges = data["hyperedges"]["paper_author"]
        hyperedges = [
            [node_id_map[aid] for aid in aids if aid in node_id_map]
            for aids in pa_edges.values()
        ]
        # Build Hypergraph and move to device
        HG = Hypergraph(len(nodes), hyperedges).to(device)

        # Align features to hypergraph order (same as old: fill where exists)
        X = torch.zeros((len(nodes), X_np.shape[1]), dtype=torch.float32, device=device)
        for aid, row in author_index.items():
            if aid in node_id_map:
                X[node_id_map[aid]] = torch.from_numpy(X_np[row]).to(device)

        # Build model (structure identical; input_dim from original X shape)
        model = HGNN_SI(
            input_dim=X_np.shape[1],
            hidden_dim=hidden_dim,
            out_dim=64,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_bn=True,
            use_residual=True,
        ).to(device)

        # Train and get embeddings
        embed = train_model(model, X, HG, epochs=epochs, lr=lr, num_samples=num_samples)

        # Save window outputs (same filenames semantics as old)
        out_embed_path = os.path.join(base_path, out_cfg["e_vec"].format(w))
        out_index_path = os.path.join(base_path, out_cfg["e_idx"].format(w))
        np.save(out_embed_path, embed)
        with open(out_index_path, "w", encoding="utf-8") as f:
            # Save AID -> HG row index (same as old)
            json.dump({aid: node_id_map[aid] for aid in author_index if aid in node_id_map}, f)
        print(f"[Saved] {out_embed_path}")
        print(f"[Saved] {out_index_path}")

    # Concatenate to [N, 5, 64] with first-window author order as base (same as old)
    if all_index_base is None:
        print("[Info] No window processed; skip concatenation.")
        return

    print("\n[Concat] Building author embedding sequence ...")
    N, D = len(all_index_base), 64
    embed_seq = []
    for w in windows:
        embed_path = os.path.join(base_path, out_cfg["e_vec"].format(w))
        index_path = os.path.join(base_path, out_cfg["e_idx"].format(w))
        if os.path.exists(embed_path) and os.path.exists(index_path):
            E = np.load(embed_path)  # [Nhg_w, 64]
            with open(index_path, "r", encoding="utf-8") as f:
                idx_map = json.load(f)  # AID -> HG row index for window w
            aligned = np.zeros((N, D), dtype=np.float32)
            for i, aid in enumerate(all_index_base):
                j = idx_map.get(aid)
                if j is not None:
                    aligned[i] = E[int(j)]
            embed_seq.append(aligned)
        else:
            print(f"[Warn] {w} missing at concat stage; fill zeros.")
            embed_seq.append(np.zeros((N, D), dtype=np.float32))

    X_embed = np.stack(embed_seq, axis=1)  # [N, 5, 64]
    seq_out_path = os.path.join(base_path, out_cfg["es_vec"])
    idx_out_path = os.path.join(base_path, out_cfg["es_idx"])
    Path(os.path.dirname(seq_out_path)).mkdir(parents=True, exist_ok=True)
    np.save(seq_out_path, X_embed)
    with open(idx_out_path, "w", encoding="utf-8") as f:
        json.dump({aid: i for i, aid in enumerate(all_index_base)}, f, indent=2)
    print(f"[Saved] {seq_out_path}")
    print(f"[Saved] {idx_out_path}")


if __name__ == "__main__":
    main()

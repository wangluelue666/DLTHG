import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os

from dhg.nn import HGNNPConv


class HGNN_SI(nn.Module):
    """
    Hypergraph Neural Network with Structural Influence (HGNN_SI)

    Core features:
        - Multi-layer HGNNPConv
        - Multi-head linear attention
        - Residual connection (optional)
        - BatchNorm (optional)
        - Dropout

    Notes:
        - hidden_dim * num_heads is the concatenated dimension for each layer
        - The last projection reduces to out_dim
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.5,
        use_bn: bool = True,
        use_residual: bool = True,
    ):
        super(HGNN_SI, self).__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.dropout = dropout

        self.hgnn_layers = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        in_dim = input_dim
        for _ in range(num_layers):
            conv = HGNNPConv(in_dim, hidden_dim)
            self.hgnn_layers.append(conv)

            attn_heads = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_heads)])
            self.attn_layers.append(attn_heads)

            if use_bn:
                self.bn_layers.append(nn.BatchNorm1d(hidden_dim * num_heads))
            in_dim = hidden_dim * num_heads

        self.fc_out = nn.Linear(in_dim, out_dim)

    def forward(self, x, hg):
        out = x
        for i in range(self.num_layers):
            h = self.hgnn_layers[i](out, hg)

            attn_outs = [head(h) for head in self.attn_layers[i]]
            h_cat = torch.cat(attn_outs, dim=-1)

            if self.use_residual and h_cat.shape == out.shape:
                h_cat = h_cat + out

            if self.use_bn:
                h_cat = self.bn_layers[i](h_cat)

            h_cat = F.relu(h_cat)
            h_cat = F.dropout(h_cat, p=self.dropout, training=self.training)

            out = h_cat

        out = self.fc_out(out)
        return out

def load_hgnn_si_from_config(dataset: str, target: str, config_path: str):
    """
    Load HGNN_SI hyperparameters from hyperparameter.yaml.

    Supports both layouts:
      A) feature_module -> hg_embed -> {dataset} -> {target} -> params
      B) {dataset} -> {target} -> params
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        hp_config = yaml.safe_load(f)

    # descend if using the nested layout
    base = hp_config
    if isinstance(base, dict) and "feature_module" in base:
        base = base["feature_module"]
    if isinstance(base, dict) and "hg_embed" in base:
        base = base["hg_embed"]

    if dataset not in base:
        raise KeyError(
            f"No hyperparameter section for dataset '{dataset}'. "
            f"Available: {list(base.keys())}"
        )
    if target not in base[dataset]:
        raise KeyError(
            f"No hyperparameter setting found for target '{target}' under dataset '{dataset}'. "
            f"Available: {list(base[dataset].keys())}"
        )

    params = base[dataset][target]

    return {
        "hidden_dim": int(params.get("hidden_dim", 128)),
        "num_heads": int(params.get("num_heads", 2)),
        "num_layers": int(params.get("num_layers", 2)),
        "dropout": float(params.get("dropout", 0.5)),
        "use_bn": bool(params.get("use_bn", True)),
        "use_residual": bool(params.get("use_residual", True)),
        "lr": float(params.get("lr", 1e-3)),
        "epochs": int(params.get("epochs", 100)),
        "num_samples": int(params.get("num_samples", 1024)),
    }

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os


# Utility: Load Hyperparameters
def load_hyperparams(dataset, target, config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["predict_module"][dataset][target]


# LSTM Encoder
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = AttentionGuidedDropout(hidden_dim, base_dropout=dropout)

    def forward(self, x):
        _, (h_n, c_n) = self.lstm(x)
        h, c = h_n[-1], c_n[-1]
        return self.dropout(h), self.dropout(c)


# Attention-Guided Dropout
class AttentionGuidedDropout(nn.Module):
    def __init__(self, hidden_dim, base_dropout=0.2):
        super().__init__()
        self.base_dropout = base_dropout
        self.attn_fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        attn_score = torch.sigmoid(self.attn_fc(h))  # (B, 1)
        scaled_dropout = self.base_dropout * (1.0 - attn_score)
        mask = (torch.rand_like(h) > scaled_dropout).float()
        return h * mask / (1.0 - scaled_dropout + 1e-8)


# LSTM Decoder
class LSTMDecoder(nn.Module):
    def __init__(self, hidden_dim=128, pred_len=6, base_dropout=0.2):
        super().__init__()
        self.pred_len = pred_len
        self.lstm_cell = nn.LSTMCell(1, hidden_dim)
        self.attn_dropout = AttentionGuidedDropout(hidden_dim, base_dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, init_input, h, c):
        x = init_input
        outputs = []
        for _ in range(self.pred_len):
            h, c = self.lstm_cell(x, (h, c))
            h_drop = self.attn_dropout(h)
            x = self.fc(h_drop)
            outputs.append(x)
        return torch.cat(outputs, dim=1)


# Gated Fusion
class GatedFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out_fc = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, h_seq, h_graph):
        concat = torch.cat([h_seq, h_graph], dim=-1)
        gate = torch.sigmoid(self.gate_fc(concat))
        fused = gate * h_seq + (1 - gate) * h_graph
        out = self.out_fc(fused)
        return self.ln(out + fused)


# Dual LSTM Predictor
class DualLSTMPredictor(nn.Module):
    def __init__(self, dataset, target,
                 seq_input_dim=3, graph_input_dim=64,
                 pred_len=6,
                 config_path="./config/hyperparameter.yaml"):
        super().__init__()
        hp = load_hyperparams(dataset, target, config_path)

        hidden_dim = hp["hidden_dim"]
        num_layers = hp["num_layers"]
        dropout = hp["dropout"]

        self.encoder_seq = LSTMEncoder(seq_input_dim, hidden_dim, num_layers, dropout=dropout)
        self.encoder_graph = LSTMEncoder(graph_input_dim, hidden_dim, num_layers, dropout=dropout)
        self.fusion = GatedFusion(hidden_dim)
        self.decoder = LSTMDecoder(hidden_dim, pred_len, base_dropout=dropout)

        self.lr = hp["lr"]
        self.batch_size = hp["batch_size"]
        self.max_epochs = hp["max_epochs"]

    def forward(self, x_seq, x_graph):
        h_seq, _ = self.encoder_seq(x_seq)
        h_graph, _ = self.encoder_graph(x_graph)
        h = self.fusion(h_seq, h_graph)
        c = torch.zeros_like(h)
        init_input = torch.zeros(x_seq.size(0), 1, device=x_seq.device)
        return self.decoder(init_input, h, c)

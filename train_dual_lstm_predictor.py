import argparse
import json
import os
import time
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from models.dual_lstm_predictor import DualLSTMPredictor


# Metrics
def regression_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = (y_true >= 0) & (y_pred >= 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {k: float("nan") for k in ["MALE", "RMSLE", "logR2", "MAE", "RMSE", "R2"]}

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    return {
        "MALE": float(np.mean(np.abs(log_pred - log_true))),
        "RMSLE": float(np.sqrt(np.mean((log_pred - log_true) ** 2))),
        "logR2": float(1 - np.sum((log_true - log_pred) ** 2) / (np.sum((log_true - np.mean(log_true)) ** 2) + 1e-8)),
        "MAE": float(np.mean(np.abs(y_pred - y_true))),
        "RMSE": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "R2": float(1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-8))
    }


def evaluate_all_years(Y_true, Y_pred):
    year_metrics = {}
    for year in range(Y_true.shape[1]):
        year_metrics[f"year_{2015 + year}"] = regression_metrics(Y_true[:, year], Y_pred[:, year])
    metrics_agg = regression_metrics(Y_true.flatten(), Y_pred.flatten())
    year_metrics["overall"] = metrics_agg
    return year_metrics


def predict_in_batches(model, data_tuple, device, batch_size=512):
    model.eval()
    results = []
    x_seq, x_graph = data_tuple
    with torch.no_grad():
        for i in range(0, x_seq.size(0), batch_size):
            x_seq_batch = x_seq[i:i + batch_size].to(device)
            x_graph_batch = x_graph[i:i + batch_size].to(device)
            out = model(x_seq_batch, x_graph_batch).cpu()
            results.append(out)
    return torch.cat(results, dim=0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["acm", "aps", "dblp"])
    parser.add_argument("--target", type=str, required=True, choices=["paper", "citation", "hindex"])
    parser.add_argument("--eval_mode", type=str, default="cumulate", choices=["cumulate"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu")

    # load path config
    with open("./config/path.yaml", "r") as f:
        path_config = yaml.safe_load(f)
    base_path = path_config["base_path"]

    # input paths
    seq_vec_path = os.path.join(base_path, path_config["features"]["sequence"][args.dataset][args.target]["seq_vec"])
    embed_vec_path = os.path.join(base_path, path_config["features"]["hg_embed"][args.dataset][args.target]["es_vec"])
    target_path = os.path.join(base_path, path_config["target_path"][args.eval_mode][args.dataset])
    result_path = os.path.join(base_path, path_config["result_path"][args.eval_mode][args.dataset][args.target])

    # load data
    X_seq = np.load(seq_vec_path)
    X_graph = np.load(embed_vec_path)
    Y_all = np.load(target_path)
    target_idx = {"paper": 0, "citation": 1, "hindex": 2}[args.target]
    Y = Y_all[:, :, target_idx]

    # split
    X_seq_train, X_seq_temp, X_graph_train, X_graph_temp, Y_train, Y_temp = train_test_split(
        X_seq, X_graph, Y, test_size=0.3, random_state=args.seed)
    X_seq_val, X_seq_test, X_graph_val, X_graph_test, Y_val, Y_test = train_test_split(
        X_seq_temp, X_graph_temp, Y_temp, test_size=0.5, random_state=args.seed)

    def pack(x1, x2, y):
        return torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    X_seq_train, X_graph_train, Y_train = pack(X_seq_train, X_graph_train, Y_train)
    X_seq_val, X_graph_val, Y_val = pack(X_seq_val, X_graph_val, Y_val)
    X_seq_test, X_graph_test, Y_test = pack(X_seq_test, X_graph_test, Y_test)

    # model & hyperparams
    model = DualLSTMPredictor(dataset=args.dataset, target=args.target).to(device)
    optimizer = optim.Adam(model.parameters(), lr=model.lr)
    loss_fn = nn.MSELoss()

    best_val_r2 = -1
    best_epoch = 0
    best_pred = {}
    start_time = time.time()

    for epoch in range(model.max_epochs):
        model.train()
        permutation = torch.randperm(X_seq_train.size(0))
        total_loss = 0.0

        for i in range(0, X_seq_train.size(0), model.batch_size):
            idx = permutation[i:i + model.batch_size]
            x_seq_batch = X_seq_train[idx].to(device)
            x_graph_batch = X_graph_train[idx].to(device)
            y_batch = Y_train[idx].to(device)

            pred = model(x_seq_batch, x_graph_batch)
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_seq_batch.size(0)

        print(f"Epoch {epoch + 1:3d}/{model.max_epochs}, Loss: {total_loss / X_seq_train.size(0):.6f}")

        # validation
        model.eval()
        with torch.no_grad():
            val_pred = predict_in_batches(model, (X_seq_val, X_graph_val), device, model.batch_size)
            val_r2 = regression_metrics(Y_val.numpy().flatten(), val_pred.flatten())["R2"]
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_epoch = epoch
                best_pred["train"] = predict_in_batches(model, (X_seq_train, X_graph_train), device, model.batch_size)
                best_pred["val"] = val_pred
                best_pred["test"] = predict_in_batches(model, (X_seq_test, X_graph_test), device, model.batch_size)
            elif epoch - best_epoch >= args.patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {args.patience} epochs)")
                break

    train_time = time.time() - start_time

    # only save metrics result
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    results = {
        "train": evaluate_all_years(Y_train.numpy(), best_pred["train"]),
        "val": evaluate_all_years(Y_val.numpy(), best_pred["val"]),
        "test": evaluate_all_years(Y_test.numpy(), best_pred["test"]),
        "time_used": {"train_time": {"seconds": round(train_time, 2)}}
    }
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n Save the results to: {result_path}")
    for split, yearly_metrics in results.items():
        if split == "time_used":
            continue
        print(f"\n{'=' * 30}\n{split.upper()} SET:")
        for year, metrics in yearly_metrics.items():
            print(f"  {year}:")
            for k, v in metrics.items():
                print(f"    {k:8s}: {v:.4f}")


if __name__ == "__main__":
    main()

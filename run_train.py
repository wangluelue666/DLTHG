"""
run_train.py

This script executes the training phase for author influence prediction.
It includes:
1. Extracting sequence features
2. Extracting embedding features
3. Training the Dual-LSTM predictor

Usage:
    python run_train.py --dataset acm --target hindex --gpu_id 3
"""

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Training phase for influence prediction")
    parser.add_argument("--dataset", type=str, required=True, choices=["acm", "aps", "dblp"], help="Dataset name")
    parser.add_argument("--target", type=str, required=True, choices=["paper", "citation", "hindex"], help="Prediction target")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    args = parser.parse_args()

    commands = [
        f"python extract_seq_features.py --dataset {args.dataset} --target {args.target}",
        f"python extract_embed_features.py --dataset {args.dataset} --target {args.target} --gpu_id {args.gpu_id}",
        f"python train_dual_lstm_predictor.py --dataset {args.dataset} --target {args.target} --gpu_id {args.gpu_id}"
    ]

    for cmd in commands:
        print(f"\n>>> Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()

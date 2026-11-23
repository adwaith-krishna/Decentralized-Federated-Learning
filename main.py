"""
Full Decentralized Federated Learning Pipeline.
- Dataset: MNIST (70,000 samples)
- Split: 70% Train (49,000), 15% Validation (10,500), 15% Test (10,500)
- Distribution: I.I.D. across 50 clients (980 train, 210 val, 210 test each)
- Model: Custom 2-Layer CNN
- Network: Dynamic P2P topology (neighbors change each iteration)
- Aggregation: Decentralized FedAvg
- Execution: Fully parallel local training and aggregation
- Stopping: Early stopping when best validation performance converges
"""

import os
import sys
import json
import time
import argparse
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from implementation.dataset import get_federated_mnist_dataloaders
from implementation.client import Client
from implementation.topology import DynamicTopology
from implementation.engine import ParallelDecentralizedEngine
from implementation.early_stopping import EarlyStopping
from implementation.aggregation import decentralized_fedavg

def parse_args():
    parser = argparse.ArgumentParser(description="Decentralized Federated Learning on MNIST")
    parser.add_argument("--num_clients", type=int, default=50, help="Number of decentralized clients")
    parser.add_argument("--neighbors", type=int, default=4, help="Number of peer neighbors per client (k)")
    parser.add_argument("--max_rounds", type=int, default=25, help="Maximum communication rounds")
    parser.add_argument("--local_epochs", type=int, default=1, help="Local epochs per round per client")
    parser.add_argument("--batch_size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=0.02, help="Client learning rate")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience rounds")
    parser.add_argument("--min_delta", type=float, default=0.002, help="Early stopping minimum delta")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_decentralized_model.pt", help="Checkpoint save path")
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 70)
    print("      DECENTRALIZED FEDERATED LEARNING - MNIST EXPERIMENT")
    print("=" * 70)
    print(f"Clients           : {args.num_clients}")
    print(f"Dataset Split     : 70% Train, 15% Val, 15% Test (I.I.D.)")
    print(f"Dynamic Neighbors : k = {args.neighbors} (changes every iteration)")
    print(f"Aggregation       : Decentralized FedAvg")
    print(f"Execution         : Fully Parallel Concurrency")
    print(f"Early Stopping    : Patience = {args.patience} rounds (min delta = {args.min_delta})")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1/5] Hardware setup: execution device = {device}")

    # 1. Load Data
    print(f"[2/5] Loading MNIST and partitioning across {args.num_clients} clients (I.I.D)...")
    client_loaders, global_loaders = get_federated_mnist_dataloaders(
        dataset_dir="Dataset",
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        seed=42
    )

    # 2. Instantiate Clients
    print(f"[3/5] Instantiating {args.num_clients} 2-Layer CNN clients...")
    clients = {}
    for cid in range(args.num_clients):
        clients[cid] = Client(
            client_id=cid,
            train_loader=client_loaders[cid]["train"],
            val_loader=client_loaders[cid]["val"],
            test_loader=client_loaders[cid]["test"],
            device=device,
            lr=args.lr
        )

    # 3. Setup Dynamic Topology, Parallel Engine, and Early Stopping
    topology = DynamicTopology(
        num_clients=args.num_clients,
        neighbors_per_client=args.neighbors,
        symmetric=True,
        seed=42
    )
    engine = ParallelDecentralizedEngine(clients=clients, topology=topology)
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        mode="max",
        checkpoint_path=args.checkpoint
    )

    print(f"[4/5] Starting Decentralized Training (Up to {args.max_rounds} rounds, stopping at best model)...")
    print("-" * 70)
    print(f"{'Round':<7} | {'Train Loss':<11} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10} | {'Patience':<8} | {'Time (s)':<8}")
    print("-" * 70)

    history = {
        "round": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "round_time": []
    }

    start_total_time = time.time()

    for r in range(args.max_rounds):
        round_stats = engine.run_round(round_num=r, local_epochs=args.local_epochs)
        
        train_loss = round_stats["avg_train_loss"]
        train_acc = round_stats["avg_train_acc"]
        val_loss = round_stats["avg_val_loss"]
        val_acc = round_stats["avg_val_acc"]
        r_time = round_stats["round_duration"]

        history["round"].append(r)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["round_time"].append(r_time)

        # Consensus weights for tracking global best model
        current_round_weights = [clients[i].get_weights() for i in range(args.num_clients)]
        consensus_weights = decentralized_fedavg(current_round_weights[0], current_round_weights[1:])

        is_best = early_stopping.step(metric=val_acc, round_num=r, model_weights=consensus_weights)
        best_marker = " ★ (Best)" if is_best else ""
        patience_str = f"{early_stopping.counter}/{args.patience}"

        print(f"R{r+1:<5} | {train_loss:<11.4f} | {train_acc*100:<9.2f}% | {val_loss:<10.4f} | {val_acc*100:<9.2f}% | {patience_str:<8} | {r_time:<6.2f}s{best_marker}")

        if early_stopping.early_stop:
            print("-" * 70)
            print(f"✓ Early stopping triggered at Round {r+1}! Best model was obtained at Round {early_stopping.best_round+1} with Val Acc: {early_stopping.best_score*100:.2f}%.")
            break

    total_training_time = time.time() - start_total_time
    print(f"\nTotal decentralized training duration: {total_training_time:.2f}s")

    # 4. Final Evaluation on 15% Held-Out Test Set
    print(f"\n[5/5] Performing Final Evaluation on 15% Test Set with Best Model...")
    best_weights = early_stopping.load_best_weights()

    # Load best weights into all clients and evaluate on test set in parallel
    for cid in range(args.num_clients):
        clients[cid].set_weights(best_weights)

    test_results = engine.parallel_evaluate(dataset_type="test")
    avg_test_loss = sum(r[0] for r in test_results.values()) / args.num_clients
    avg_test_acc = sum(r[1] for r in test_results.values()) / args.num_clients

    print("=" * 70)
    print("                  FINAL TEST EVALUATION RESULTS")
    print("=" * 70)
    print(f"Best Model Round       : Round {early_stopping.best_round + 1}")
    print(f"Best Validation Accuracy: {early_stopping.best_score * 100:.2f}%")
    print(f"Average Test Accuracy  : {avg_test_acc * 100:.2f}% across all 50 clients")
    print(f"Average Test Loss      : {avg_test_loss:.4f}")
    print(f"Best Model Saved To    : {early_stopping.checkpoint_path}")
    print("=" * 70)

    # 5. Save History & Plot Curves
    os.makedirs("results", exist_ok=True)
    with open("results/training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history["round"], history["train_loss"], label="Train Loss", marker="o")
    plt.plot(history["round"], history["val_loss"], label="Val Loss", marker="s")
    plt.axvline(x=early_stopping.best_round, color="r", linestyle="--", label=f"Best Round ({early_stopping.best_round+1})")
    plt.title("Decentralized Training & Validation Loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history["round"], [a * 100 for a in history["train_acc"]], label="Train Acc", marker="o")
    plt.plot(history["round"], [a * 100 for a in history["val_acc"]], label="Val Acc", marker="s")
    plt.axvline(x=early_stopping.best_round, color="r", linestyle="--", label=f"Best Round ({early_stopping.best_round+1})")
    plt.title("Decentralized Training & Validation Accuracy (%)")
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plot_path = "results/training_metrics.png"
    plt.savefig(plot_path, dpi=200)
    print(f"Saved training history to results/training_history.json")
    print(f"Saved metric curves to {plot_path}")

if __name__ == "__main__":
    main()

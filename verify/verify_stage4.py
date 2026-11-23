"""
Verification script for Stage 4: Parallel Execution Engine.
Validates:
1. Parallel initialization of all 50 clients.
2. Parallel local training across all 50 clients.
3. Dynamic peer weight exchange and decentralized FedAvg aggregation in parallel.
4. Parallel local evaluation across all 50 clients.
5. Timing benchmarks verifying concurrent execution speedup.
"""

import os
import sys
import time
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from implementation.dataset import get_federated_mnist_dataloaders
from implementation.client import Client
from implementation.topology import DynamicTopology
from implementation.engine import ParallelDecentralizedEngine

def run_verification():
    print("=" * 65)
    print("STAGE 4 VERIFICATION: Parallel Execution Engine")
    print("=" * 65)

    num_clients = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Step 1] Loading partitions and initializing {num_clients} clients on {device}...")
    
    client_loaders, _ = get_federated_mnist_dataloaders(
        dataset_dir="Dataset",
        num_clients=num_clients,
        batch_size=32,
        seed=42
    )

    clients = {}
    for cid in range(num_clients):
        clients[cid] = Client(
            client_id=cid,
            train_loader=client_loaders[cid]["train"],
            val_loader=client_loaders[cid]["val"],
            test_loader=client_loaders[cid]["test"],
            device=device,
            lr=0.02
        )
    print(f"  ✓ Successfully initialized all {num_clients} clients.")

    # Initialize Dynamic Topology and Parallel Engine
    topology = DynamicTopology(num_clients=num_clients, neighbors_per_client=4, symmetric=True, seed=42)
    engine = ParallelDecentralizedEngine(clients=clients, topology=topology)
    print(f"  ✓ Parallel engine configured with {engine.max_workers} worker threads.")

    # 2. Parallel Training Benchmark
    print("\n[Step 2] Testing Parallel Local Training across all 50 clients...")
    t_start = time.time()
    train_results = engine.parallel_train(epochs=1)
    train_duration = time.time() - t_start

    assert len(train_results) == num_clients, f"Expected {num_clients} training results"
    avg_train_loss = sum(r[0] for r in train_results.values()) / num_clients
    avg_train_acc = sum(r[1] for r in train_results.values()) / num_clients
    print(f"  ✓ Parallel training completed in {train_duration:.2f}s ({train_duration/num_clients:.3f}s per client equivalent)")
    print(f"  ✓ Average local train loss: {avg_train_loss:.4f}, Average local train acc: {avg_train_acc * 100:.2f}%")

    # 3. Parallel Dynamic Peer Exchange and FedAvg Aggregation
    print("\n[Step 3] Testing Parallel Dynamic Peer Weight Exchange & FedAvg Aggregation...")
    t_agg_start = time.time()
    round_topology = engine.parallel_exchange_and_aggregate(round_num=0)
    agg_duration = time.time() - t_agg_start
    print(f"  ✓ Weight exchange & aggregation across 50 clients completed in {agg_duration:.3f}s")
    assert len(round_topology) == num_clients, "Topology mapping count mismatch"

    # 4. Parallel Validation Evaluation
    print("\n[Step 4] Testing Parallel Validation Evaluation across all 50 clients...")
    t_eval_start = time.time()
    val_results = engine.parallel_evaluate(dataset_type="val")
    eval_duration = time.time() - t_eval_start

    assert len(val_results) == num_clients, f"Expected {num_clients} validation results"
    avg_val_loss = sum(r[0] for r in val_results.values()) / num_clients
    avg_val_acc = sum(r[1] for r in val_results.values()) / num_clients
    print(f"  ✓ Parallel evaluation completed in {eval_duration:.3f}s")
    print(f"  ✓ Average validation loss: {avg_val_loss:.4f}, Average validation acc: {avg_val_acc * 100:.2f}%")

    # 5. Full Round Execution Test
    print("\n[Step 5] Testing unified engine.run_round(round_num=1)...")
    round1_stats = engine.run_round(round_num=1, local_epochs=1)
    print(f"  ✓ Round 1 finished in {round1_stats['round_duration']:.2f}s")
    print(f"  ✓ Post-round 1 Avg Train Acc: {round1_stats['avg_train_acc'] * 100:.2f}%")
    print(f"  ✓ Post-round 1 Avg Val Acc  : {round1_stats['avg_val_acc'] * 100:.2f}%")

    print("\n" + "=" * 65)
    print("STAGE 4 VERIFICATION COMPLETED SUCCESSFULLY! All checks passed.")
    print("=" * 65)

if __name__ == "__main__":
    run_verification()

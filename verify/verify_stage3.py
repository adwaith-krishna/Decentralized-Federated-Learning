"""
Verification script for Stage 3: Dynamic Topology & Decentralized FedAvg.
Validates:
1. Dynamic communication graph generation across 50 clients.
2. Self-connection exclusion (no client communicates with itself).
3. Dynamic topology evolution: neighbor set changes across consecutive rounds.
4. Mathematical correctness of decentralized FedAvg aggregation with known synthetic tensors.
5. Integration test: decentralized FedAvg on real TwoLayerCNN models.
"""

import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from implementation.topology import DynamicTopology
from implementation.aggregation import decentralized_fedavg
from implementation.model import TwoLayerCNN

def run_verification():
    print("=" * 65)
    print("STAGE 3 VERIFICATION: Dynamic Topology & Decentralized FedAvg")
    print("=" * 65)

    # 1. Dynamic Topology Generation
    print("\n[Step 1] Initializing Dynamic Topology for 50 clients...")
    num_clients = 50
    k_neighbors = 4
    topology = DynamicTopology(
        num_clients=num_clients,
        neighbors_per_client=k_neighbors,
        symmetric=True,
        seed=42
    )

    g_round0 = topology.get_neighbors_for_round(round_num=0)
    g_round1 = topology.get_neighbors_for_round(round_num=1)
    g_round2 = topology.get_neighbors_for_round(round_num=2)

    print(f"  ✓ Total clients in network: {len(g_round0)}")
    assert len(g_round0) == num_clients, f"Expected {num_clients} clients in graph"

    # 2. Check no self-connections and valid degree
    print("\n[Step 2] Validating topology properties (no self-loops, degree bounds)...")
    for cid in range(num_clients):
        neighbors = g_round0[cid]
        assert cid not in neighbors, f"Client {cid} connected to itself!"
        assert len(neighbors) >= k_neighbors, f"Client {cid} has fewer than {k_neighbors} neighbors"

    # Display sample connections for first 5 clients
    for cid in range(5):
        print(f"  Client {cid:2d} Round 0 neighbors: {g_round0[cid]}")
    print("  ✓ No self-connections detected and valid neighbor degrees maintained.")

    # 3. Verify Dynamic Change across rounds
    print("\n[Step 3] Checking neighbor changes across consecutive iterations...")
    change_r0_r1 = topology.verify_dynamic_change(0, 1)
    change_r1_r2 = topology.verify_dynamic_change(1, 2)
    print(f"  ✓ Neighbor change fraction between Round 0 and Round 1: {change_r0_r1 * 100:.2f}%")
    print(f"  ✓ Neighbor change fraction between Round 1 and Round 2: {change_r1_r2 * 100:.2f}%")
    assert change_r0_r1 > 0.50, f"Expected >50% neighbor change between rounds, got {change_r0_r1}"
    assert change_r1_r2 > 0.50, f"Expected >50% neighbor change between rounds, got {change_r1_r2}"
    print("  ✓ Topology dynamically changes across every iteration.")

    # 4. Mathematical Exactness of FedAvg Aggregation
    print("\n[Step 4] Testing FedAvg mathematical correctness on synthetic tensors...")
    # Synthetic test: Client A (weights = [1.0, 2.0]), Neighbor B ([3.0, 4.0]), Neighbor C ([5.0, 6.0])
    # Expected average = [(1 + 3 + 5)/3, (2 + 4 + 6)/3] = [3.0, 4.0]
    local_weights = {"param": torch.tensor([1.0, 2.0])}
    neighbor_weights = [
        {"param": torch.tensor([3.0, 4.0])},
        {"param": torch.tensor([5.0, 6.0])}
    ]
    avg_result = decentralized_fedavg(local_weights, neighbor_weights)
    expected = torch.tensor([3.0, 4.0])
    diff = (avg_result["param"] - expected).abs().max().item()
    print(f"  ✓ Aggregated tensor: {avg_result['param'].tolist()}, Expected: {expected.tolist()}")
    assert diff < 1e-6, f"FedAvg aggregation mathematical test failed with diff {diff}"
    print("  ✓ Exact mathematical FedAvg confirmed.")

    # 5. Integration Test with TwoLayerCNN Models
    print("\n[Step 5] Integration test: Aggregating real TwoLayerCNN models...")
    model_i = TwoLayerCNN()
    model_j1 = TwoLayerCNN()
    model_j2 = TwoLayerCNN()

    w_i = model_i.state_dict()
    w_j1 = model_j1.state_dict()
    w_j2 = model_j2.state_dict()

    agg_weights = decentralized_fedavg(w_i, [w_j1, w_j2])
    assert set(agg_weights.keys()) == set(w_i.keys()), "Aggregated weights keys mismatch"

    # Check conv1.weight manually
    expected_conv1 = (w_i["conv1.weight"] + w_j1["conv1.weight"] + w_j2["conv1.weight"]) / 3.0
    conv1_diff = (agg_weights["conv1.weight"] - expected_conv1).abs().max().item()
    assert conv1_diff < 1e-6, f"Conv1 weights aggregation discrepancy: {conv1_diff}"
    print(f"  ✓ Conv1 weight aggregation discrepancy: {conv1_diff:.2e} (Passed)")

    print("\n" + "=" * 65)
    print("STAGE 3 VERIFICATION COMPLETED SUCCESSFULLY! All checks passed.")
    print("=" * 65)

if __name__ == "__main__":
    run_verification()

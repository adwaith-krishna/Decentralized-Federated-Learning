"""
Verification script for Stage 5: Early Stopping, Checkpointing & Final Test Evaluation.
Validates:
1. EarlyStopping behavior on simulated metric improvements and plateauing.
2. Best model checkpoint saving and loading from disk.
3. Test evaluation on the 15% test partition.
4. Correct termination when patience runs out.
"""

import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from implementation.early_stopping import EarlyStopping
from implementation.model import TwoLayerCNN
from implementation.dataset import get_federated_mnist_dataloaders
from implementation.client import Client

def run_verification():
    print("=" * 65)
    print("STAGE 5 VERIFICATION: Early Stopping & Test Evaluation")
    print("=" * 65)

    checkpoint_test_path = "checkpoints/verify_best_model.pt"
    if os.path.exists(checkpoint_test_path):
        os.remove(checkpoint_test_path)

    # 1. Early Stopping Simulation
    print("\n[Step 1] Testing EarlyStopping logic (patience=3, mode='max')...")
    stopper = EarlyStopping(patience=3, min_delta=0.005, mode="max", checkpoint_path=checkpoint_test_path)
    
    dummy_model = TwoLayerCNN()
    weights = dummy_model.state_dict()

    # Round 0: 70.0% -> new best
    assert stopper.step(0.70, round_num=0, model_weights=weights) == True
    assert stopper.counter == 0 and stopper.early_stop == False
    print("  ✓ Round 0: Acc 70.0% (New best registered)")

    # Round 1: 75.0% -> new best
    assert stopper.step(0.75, round_num=1, model_weights=weights) == True
    assert stopper.counter == 0 and stopper.early_stop == False
    print("  ✓ Round 1: Acc 75.0% (New best registered)")

    # Round 2: 75.1% -> within min_delta (0.005), so not considered improvement
    assert stopper.step(0.751, round_num=2, model_weights=weights) == False
    assert stopper.counter == 1 and stopper.early_stop == False
    print("  ✓ Round 2: Acc 75.1% (Plateau counter = 1)")

    # Round 3: 74.8% -> no improvement
    assert stopper.step(0.748, round_num=3, model_weights=weights) == False
    assert stopper.counter == 2 and stopper.early_stop == False
    print("  ✓ Round 3: Acc 74.8% (Plateau counter = 2)")

    # Round 4: 74.9% -> patience reached (counter=3 >= patience=3)
    assert stopper.step(0.749, round_num=4, model_weights=weights) == False
    assert stopper.counter == 3 and stopper.early_stop == True
    print("  ✓ Round 4: Acc 74.9% (Plateau counter = 3 -> Early Stop Triggered!)")

    # 2. Checkpoint Persistence Check
    print("\n[Step 2] Validating saved checkpoint on disk...")
    assert os.path.exists(checkpoint_test_path), "Checkpoint file was not created on disk"
    loaded_weights = stopper.load_best_weights()
    assert set(loaded_weights.keys()) == set(weights.keys()), "Loaded state_dict keys mismatch"
    print(f"  ✓ Checkpoint successfully persisted and verified at {checkpoint_test_path}.")
    print(f"  ✓ Best recorded accuracy: {stopper.best_score * 100:.2f}% at Round {stopper.best_round}")

    # 3. Test Evaluation Pipeline
    print("\n[Step 3] Validating Evaluation Pipeline with Real Test Split...")
    client_loaders, global_loaders = get_federated_mnist_dataloaders(
        dataset_dir="Dataset",
        num_clients=50,
        batch_size=32,
        seed=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_client = Client(
        client_id=0,
        train_loader=client_loaders[0]["train"],
        val_loader=client_loaders[0]["val"],
        test_loader=client_loaders[0]["test"],
        device=device
    )
    eval_client.set_weights(loaded_weights)

    test_loss, test_acc = eval_client.evaluate(dataset_type="test")
    print(f"  ✓ Client 0 test evaluation passed: Loss={test_loss:.4f}, Accuracy={test_acc * 100:.2f}%")

    # Clean up test checkpoint
    if os.path.exists(checkpoint_test_path):
        os.remove(checkpoint_test_path)

    print("\n" + "=" * 65)
    print("STAGE 5 VERIFICATION COMPLETED SUCCESSFULLY! All checks passed.")
    print("=" * 65)

if __name__ == "__main__":
    run_verification()

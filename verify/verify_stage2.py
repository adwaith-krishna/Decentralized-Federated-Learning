"""
Verification script for Stage 2: Custom 2-Layer CNN & Client Local Training.
Validates:
1. TwoLayerCNN layer architecture and forward pass shapes.
2. Parameter count.
3. Client initialization on available device (CUDA/CPU).
4. Local training epoch: loss decreases, weights update.
5. Model evaluation on validation & test sets.
6. Weight export (get_weights) and weight loading (set_weights).
"""

import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from implementation.model import TwoLayerCNN
from implementation.client import Client
from implementation.dataset import get_federated_mnist_dataloaders

def run_verification():
    print("=" * 65)
    print("STAGE 2 VERIFICATION: 2-Layer CNN & Client Local Training")
    print("=" * 65)

    # 1. Model Architecture Check
    print("\n[Step 1] Inspecting TwoLayerCNN Architecture...")
    model = TwoLayerCNN(num_classes=10)
    print(model)

    dummy_input = torch.randn(4, 1, 28, 28)
    output = model(dummy_input)
    assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"
    print(f"  ✓ Forward pass output shape: {output.shape} (Expected: [4, 10])")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Trainable parameters: {total_params:,}")

    # 2. Client Initialization
    print("\n[Step 2] Initializing Client with real MNIST partition...")
    client_loaders, _ = get_federated_mnist_dataloaders(
        dataset_dir="Dataset",
        num_clients=50,
        batch_size=32,
        seed=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  ✓ Target execution device: {device}")

    client_0 = Client(
        client_id=0,
        train_loader=client_loaders[0]["train"],
        val_loader=client_loaders[0]["val"],
        test_loader=client_loaders[0]["test"],
        device=device,
        lr=0.02
    )
    print(f"  ✓ Client 0 created successfully on {device}.")

    # 3. Pre-training Evaluation
    pre_val_loss, pre_val_acc = client_0.evaluate(dataset_type="val")
    print(f"\n[Step 3] Client 0 Pre-training Val Loss: {pre_val_loss:.4f}, Accuracy: {pre_val_acc * 100:.2f}%")

    # 4. Local Training Epoch
    print("\n[Step 4] Running 1 Local Training Epoch for Client 0...")
    initial_weights = client_0.get_weights()
    train_loss, train_acc = client_0.train_epoch(epochs=1)
    print(f"  ✓ Epoch 1 Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc * 100:.2f}%")

    post_weights = client_0.get_weights()
    # Verify weights actually updated
    weight_diff = sum((post_weights[k] - initial_weights[k]).abs().sum().item() for k in initial_weights)
    print(f"  ✓ Cumulative weight parameter change: {weight_diff:.4f}")
    assert weight_diff > 0, "Model weights did not update during training!"

    # 5. Post-training Evaluation
    post_val_loss, post_val_acc = client_0.evaluate(dataset_type="val")
    test_loss, test_acc = client_0.evaluate(dataset_type="test")
    print(f"\n[Step 5] Client 0 Post-training Results:")
    print(f"  ✓ Post Val Loss  : {post_val_loss:.4f}, Val Accuracy : {post_val_acc * 100:.2f}%")
    print(f"  ✓ Post Test Loss : {test_loss:.4f}, Test Accuracy: {test_acc * 100:.2f}%")
    assert post_val_acc > pre_val_acc, "Expected accuracy to improve after 1 epoch"

    # 6. Weight Get/Set Verification
    print("\n[Step 6] Testing weight export and import...")
    client_0.set_weights(initial_weights)
    restored_val_loss, restored_val_acc = client_0.evaluate(dataset_type="val")
    print(f"  ✓ Restored Val Acc: {restored_val_acc * 100:.2f}% (Matches pre-train: {pre_val_acc * 100:.2f}%)")
    assert abs(restored_val_acc - pre_val_acc) < 1e-5, "Weight restoration failed!"

    print("\n" + "=" * 65)
    print("STAGE 2 VERIFICATION COMPLETED SUCCESSFULLY! All checks passed.")
    print("=" * 65)

if __name__ == "__main__":
    run_verification()

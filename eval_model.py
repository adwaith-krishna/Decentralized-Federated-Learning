"""
Evaluation & Inference script for the saved best decentralized model.
Evaluates the checkpoint on the 15% test dataset without retraining.
"""

import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from implementation.model import TwoLayerCNN
from implementation.dataset import get_federated_mnist_dataloaders

def evaluate_saved_model(checkpoint_path: str = "checkpoints/best_decentralized_model.pt"):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"EVALUATING TRAINED DECENTRALIZED MODEL ON TEST DATASET")
    print(f"Device: {device}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    print("=" * 65)

    # 1. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    best_round = checkpoint.get("round", "N/A")
    best_val_acc = checkpoint.get("metric", "N/A")

    if isinstance(best_round, int):
        best_round += 1
    if isinstance(best_val_acc, float):
        best_val_acc = f"{best_val_acc * 100:.2f}%"

    print(f"  ✓ Checkpoint converged at: Round {best_round}")
    print(f"  ✓ Checkpoint validation accuracy: {best_val_acc}")

    # 2. Instantiate Model and Load Weights
    model = TwoLayerCNN(num_classes=10).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Load 15% Test Dataset
    print("\n[Loading Test Partition (10,500 samples)...]")
    _, global_loaders = get_federated_mnist_dataloaders(
        dataset_dir="Dataset",
        num_clients=50,
        batch_size=64,
        seed=42
    )
    test_loader = global_loaders["test"]

    # 4. Evaluate
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * len(target)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total_samples += len(target)

    test_loss = total_loss / total_samples
    test_acc = correct / total_samples

    print("\n" + "=" * 65)
    print("                   TEST SET RESULTS")
    print("=" * 65)
    print(f"Total Test Samples : {total_samples:,}")
    print(f"Test Accuracy      : {test_acc * 100:.2f}%")
    print(f"Test Loss          : {test_loss:.4f}")
    print("=" * 65)

if __name__ == "__main__":
    evaluate_saved_model()

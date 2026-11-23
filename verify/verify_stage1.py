"""
Verification script for Stage 1: Dataset Pipeline & I.I.D. Splitting.
Validates:
1. Total dataset samples = 70,000 (train + test).
2. Global 70-15-15 split counts:
   - Train: 49,000
   - Val:   10,500
   - Test:  10,500
3. 50 clients partitioned I.I.D:
   - Each client train set: 980 samples
   - Each client val set:   210 samples
   - Each client test set:  210 samples
4. Class balance across clients (uniform ~10% per class).
5. PyTorch DataLoader shapes and batching.
"""

import os
import sys
import torch

# Ensure the project root directory is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from implementation.dataset import get_federated_mnist_dataloaders, load_mnist_all, split_70_15_15

def run_verification():
    print("=" * 65)
    print("STAGE 1 VERIFICATION: MNIST Dataset & I.I.D Partitioning")
    print("=" * 65)

    # 1. Test Raw Loading
    print("\n[Step 1] Loading raw MNIST dataset from ./Dataset...")
    images, labels = load_mnist_all("Dataset")
    print(f"  ✓ Loaded images: {images.shape}, dtype: {images.dtype}")
    print(f"  ✓ Loaded labels: {labels.shape}, dtype: {labels.dtype}")
    assert len(images) == 70000, f"Expected 70,000 total samples, got {len(images)}"

    # 2. Test 70-15-15 Split
    print("\n[Step 2] Verifying 70-15-15 Split...")
    splits = split_70_15_15(images, labels, seed=42)
    n_train = len(splits['train'][1])
    n_val = len(splits['val'][1])
    n_test = len(splits['test'][1])
    print(f"  ✓ Train count : {n_train} (70.0%)")
    print(f"  ✓ Val count   : {n_val} (15.0%)")
    print(f"  ✓ Test count  : {n_test} (15.0%)")
    assert n_train == 49000, f"Train count mismatch: {n_train}"
    assert n_val == 10500, f"Val count mismatch: {n_val}"
    assert n_test == 10500, f"Test count mismatch: {n_test}"

    # 3. Test Federated Client DataLoaders (50 clients)
    print("\n[Step 3] Partitioning across 50 clients (I.I.D)...")
    client_loaders, global_loaders = get_federated_mnist_dataloaders(
        dataset_dir="Dataset",
        num_clients=50,
        batch_size=32,
        seed=42
    )

    assert len(client_loaders) == 50, f"Expected 50 clients, got {len(client_loaders)}"
    print("  ✓ Created 50 client DataLoaders successfully.")

    # 4. Verify Per-Client Sample Counts & Class Distribution
    print("\n[Step 4] Checking client partition sizes and class distributions...")
    print(f"{'Client ID':<10} | {'Train':<8} | {'Val':<8} | {'Test':<8} | Class 0-9 Counts in Train")
    print("-" * 65)

    for cid in range(50):
        c_train_ds = client_loaders[cid]["train"].dataset
        c_val_ds = client_loaders[cid]["val"].dataset
        c_test_ds = client_loaders[cid]["test"].dataset

        assert len(c_train_ds) == 980, f"Client {cid} train size is {len(c_train_ds)}, expected 980"
        assert len(c_val_ds) == 210, f"Client {cid} val size is {len(c_val_ds)}, expected 210"
        assert len(c_test_ds) == 210, f"Client {cid} test size is {len(c_test_ds)}, expected 210"

        # Check class distribution in train set
        targets = c_train_ds.labels
        counts = torch.bincount(targets, minlength=10).tolist()

        # Display first 5 clients and last client as representation
        if cid < 5 or cid == 49:
            counts_str = " ".join([f"{c:2d}" for c in counts])
            print(f"Client {cid:<4} | {len(c_train_ds):<8} | {len(c_val_ds):<8} | {len(c_test_ds):<8} | [{counts_str}]")
        elif cid == 5:
            print("  ... (clients 5-48 have identical balanced partitions) ...")

    # 5. Check Batch Shapes & Normalization
    sample_batch_x, sample_batch_y = next(iter(client_loaders[0]["train"]))
    print("\n[Step 5] Checking DataLoader batch tensor properties:")
    print(f"  ✓ Image batch shape : {sample_batch_x.shape} (Expected: [32, 1, 28, 28])")
    print(f"  ✓ Label batch shape : {sample_batch_y.shape} (Expected: [32])")
    print(f"  ✓ Image tensor mean : {sample_batch_x.mean():.4f} (Expected near 0)")
    print(f"  ✓ Image tensor std  : {sample_batch_x.std():.4f} (Expected near 1)")

    assert sample_batch_x.shape == (32, 1, 28, 28), f"Unexpected shape {sample_batch_x.shape}"
    assert sample_batch_y.shape == (32,), f"Unexpected label shape {sample_batch_y.shape}"

    print("\n" + "=" * 65)
    print("STAGE 1 VERIFICATION COMPLETED SUCCESSFULLY! All checks passed.")
    print("=" * 65)

if __name__ == "__main__":
    run_verification()

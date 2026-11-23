import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MNISTTensorDataset(Dataset):
    """Custom PyTorch Dataset for MNIST tensors."""
    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        # images: (N, 28, 28) or (N, 1, 28, 28) in uint8 or float
        if images.ndim == 3:
            images = images.unsqueeze(1)  # (N, 1, 28, 28)
        
        # Convert to float and normalize with standard MNIST mean and std
        if images.dtype != torch.float32:
            images = images.float() / 255.0
        
        # Standard normalization: mean=0.1307, std=0.3081
        self.images = (images - 0.1307) / 0.3081
        self.labels = labels.long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def _read_idx_file(filepath: str) -> torch.Tensor:
    """Reads idx ubyte files using torchvision or fallback numpy."""
    try:
        from torchvision.datasets.mnist import read_image_file, read_label_file
        if "images" in filepath or "idx3" in filepath:
            return read_image_file(filepath)
        else:
            return read_label_file(filepath)
    except Exception:
        # Fallback reading
        with open(filepath, 'rb') as f:
            data = f.read()
        import struct
        magic = struct.unpack('>H', data[2:4])[0]
        if magic == 2051:  # images
            num_imgs, rows, cols = struct.unpack('>III', data[4:16])
            imgs = np.frombuffer(data[16:], dtype=np.uint8).reshape(num_imgs, rows, cols)
            return torch.from_numpy(imgs)
        elif magic == 2049:  # labels
            num_lbls = struct.unpack('>I', data[4:8])[0]
            lbls = np.frombuffer(data[8:], dtype=np.uint8)
            return torch.from_numpy(lbls)
        else:
            raise ValueError(f"Unrecognized idx format in {filepath}")


def load_mnist_all(dataset_dir: str = "Dataset"):
    """
    Loads all 70,000 MNIST images and labels from the Dataset directory.
    Combines 60,000 training and 10,000 test samples.
    """
    train_img_path = os.path.join(dataset_dir, "train-images.idx3-ubyte")
    train_lbl_path = os.path.join(dataset_dir, "train-labels.idx1-ubyte")
    test_img_path = os.path.join(dataset_dir, "t10k-images.idx3-ubyte")
    test_lbl_path = os.path.join(dataset_dir, "t10k-labels.idx1-ubyte")

    for p in [train_img_path, train_lbl_path, test_img_path, test_lbl_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required MNIST file not found at: {p}")

    train_imgs = _read_idx_file(train_img_path)
    train_lbls = _read_idx_file(train_lbl_path)
    test_imgs = _read_idx_file(test_img_path)
    test_lbls = _read_idx_file(test_lbl_path)

    all_images = torch.cat([train_imgs, test_imgs], dim=0)
    all_labels = torch.cat([train_lbls, test_lbls], dim=0)

    return all_images, all_labels


def split_70_15_15(images: torch.Tensor, labels: torch.Tensor, seed: int = 42):
    """
    Splits 70,000 total samples into:
      - 70% Train: 49,000 samples
      - 15% Validation: 10,500 samples
      - 15% Test: 10,500 samples
    Uses stratified sampling to preserve class distribution across splits.
    """
    num_samples = len(labels)
    assert num_samples == 70000, f"Expected 70,000 samples, found {num_samples}"

    rng = np.random.default_rng(seed)

    target_train = 49000
    target_val = 10500
    target_test = 10500

    classes = torch.unique(labels).numpy()
    train_indices = []
    val_indices = []
    test_indices = []

    # Proportional split per class
    for c in classes:
        c_indices = (labels == c).nonzero(as_tuple=True)[0].numpy()
        rng.shuffle(c_indices)
        
        n_c = len(c_indices)
        n_train = int(n_c * 0.70)
        n_val = int(n_c * 0.15)
        
        train_indices.extend(c_indices[:n_train])
        val_indices.extend(c_indices[n_train:n_train + n_val])
        test_indices.extend(c_indices[n_train + n_val:])

    # Reconcile integer rounding to ensure exact target counts
    rem_train = target_train - len(train_indices)
    if rem_train > 0:
        train_indices.extend(test_indices[-rem_train:])
        test_indices = test_indices[:-rem_train]
    elif rem_train < 0:
        test_indices.extend(train_indices[rem_train:])
        train_indices = train_indices[:rem_train]

    rem_val = target_val - len(val_indices)
    if rem_val > 0:
        val_indices.extend(test_indices[-rem_val:])
        test_indices = test_indices[:-rem_val]
    elif rem_val < 0:
        test_indices.extend(val_indices[rem_val:])
        val_indices = val_indices[:rem_val]

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    assert len(train_indices) == target_train, f"Train count {len(train_indices)} != {target_train}"
    assert len(val_indices) == target_val, f"Val count {len(val_indices)} != {target_val}"
    assert len(test_indices) == target_test, f"Test count {len(test_indices)} != {target_test}"

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    splits = {
        "train": (images[train_indices], labels[train_indices]),
        "val": (images[val_indices], labels[val_indices]),
        "test": (images[test_indices], labels[test_indices]),
    }
    return splits


def partition_iid(images: torch.Tensor, labels: torch.Tensor, num_clients: int = 50, seed: int = 42):
    """
    Partitions the given split across num_clients in an I.I.D. manner.
    Ensures each client receives an equal share and balanced class distribution.
    """
    rng = np.random.default_rng(seed)
    num_samples = len(labels)
    assert num_samples % num_clients == 0, f"{num_samples} not divisible by {num_clients}"
    samples_per_client = num_samples // num_clients

    classes = torch.unique(labels).numpy()
    client_indices = [[] for _ in range(num_clients)]

    # Distribute samples continuously across clients per class to maintain class balance
    client_turn = 0
    for c in classes:
        c_indices = (labels == c).nonzero(as_tuple=True)[0].numpy()
        rng.shuffle(c_indices)
        for idx in c_indices:
            client_indices[client_turn % num_clients].append(idx)
            client_turn += 1

    client_datasets = {}
    for cid in range(num_clients):
        c_idx = np.array(client_indices[cid])
        assert len(c_idx) == samples_per_client, f"Client {cid} got {len(c_idx)}, expected {samples_per_client}"
        rng.shuffle(c_idx)
        client_datasets[cid] = (images[c_idx], labels[c_idx])

    return client_datasets


def get_federated_mnist_dataloaders(
    dataset_dir: str = "Dataset",
    num_clients: int = 50,
    batch_size: int = 32,
    seed: int = 42
):
    """
    High-level factory function that loads MNIST, performs the 70-15-15 split,
    creates I.I.D partitions for all 50 clients, and returns client DataLoaders.
    
    Returns:
        client_loaders: dict mapping client_id (0..49) to:
            {
                'train': DataLoader,
                'val': DataLoader,
                'test': DataLoader
            }
        global_loaders: dict containing:
            {
                'val': DataLoader,
                'test': DataLoader
            }
    """
    all_images, all_labels = load_mnist_all(dataset_dir)
    splits = split_70_15_15(all_images, all_labels, seed=seed)

    train_partitions = partition_iid(splits["train"][0], splits["train"][1], num_clients, seed=seed)
    val_partitions = partition_iid(splits["val"][0], splits["val"][1], num_clients, seed=seed + 1)
    test_partitions = partition_iid(splits["test"][0], splits["test"][1], num_clients, seed=seed + 2)

    client_loaders = {}
    for cid in range(num_clients):
        train_ds = MNISTTensorDataset(*train_partitions[cid])
        val_ds = MNISTTensorDataset(*val_partitions[cid])
        test_ds = MNISTTensorDataset(*test_partitions[cid])

        client_loaders[cid] = {
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        }

    global_val_ds = MNISTTensorDataset(*splits["val"])
    global_test_ds = MNISTTensorDataset(*splits["test"])

    global_loaders = {
        "val": DataLoader(global_val_ds, batch_size=batch_size * 2, shuffle=False),
        "test": DataLoader(global_test_ds, batch_size=batch_size * 2, shuffle=False)
    }

    return client_loaders, global_loaders

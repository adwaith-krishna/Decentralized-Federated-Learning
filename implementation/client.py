import copy
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from implementation.model import TwoLayerCNN

class Client:
    """
    Federated Learning Client in a decentralized network.
    Maintains:
    - client_id
    - local TwoLayerCNN model
    - train, val, and test DataLoaders
    - local optimizer and loss function
    """
    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: Optional[torch.device] = None,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4
    ):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = TwoLayerCNN(num_classes=10).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

    def train_epoch(self, epochs: int = 1) -> Tuple[float, float]:
        """
        Performs local training for a given number of epochs.
        Returns:
            avg_loss: average cross-entropy loss
            accuracy: training classification accuracy (0.0 to 1.0)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for _ in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(target)
                preds = output.argmax(dim=1)
                correct += (preds == target).sum().item()
                total_samples += len(target)

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = correct / max(total_samples, 1)
        return avg_loss, accuracy

    def evaluate(self, dataset_type: str = "val") -> Tuple[float, float]:
        """
        Evaluates current local model on 'val' or 'test' set.
        Returns:
            avg_loss: average validation/test loss
            accuracy: validation/test accuracy (0.0 to 1.0)
        """
        self.model.eval()
        loader = self.val_loader if dataset_type == "val" else self.test_loader
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item() * len(target)
                preds = output.argmax(dim=1)
                correct += (preds == target).sum().item()
                total_samples += len(target)

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = correct / max(total_samples, 1)
        return avg_loss, accuracy

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Returns deepcopy of model state_dict on CPU."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Loads state_dict into client model."""
        device_state_dict = {k: v.to(self.device) for k, v in state_dict.items()}
        self.model.load_state_dict(device_state_dict)

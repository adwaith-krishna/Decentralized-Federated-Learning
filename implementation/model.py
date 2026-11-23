import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerCNN(nn.Module):
    """
    Custom 2-Layer Convolutional Neural Network for MNIST (28x28 grayscale).
    - Layer 1: Conv2d(1 -> 32, 3x3, padding=1) + ReLU + MaxPool2d(2x2) -> [B, 32, 14, 14]
    - Layer 2: Conv2d(32 -> 64, 3x3, padding=1) + ReLU + MaxPool2d(2x2) -> [B, 64, 7, 7]
    - Output Head: Linear(64 * 7 * 7 = 3136 -> 10)
    """
    def __init__(self, num_classes: int = 10):
        super(TwoLayerCNN, self).__init__()
        
        # Conv Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Linear Classifier Head
        self.fc = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Layer 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Flatten and classify
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

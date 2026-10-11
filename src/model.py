import torch
import torch.nn as nn

class MNIST_MLP(nn.Module):
    def __init__(self, input_size: int = 784, hidden1_size: int = 256, hidden2_size: int = 128, num_classes: int = 10):
        super(MNIST_MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Girdi boyutu (batch_size, 1, 28, 28) veya (batch_size, 784) olabilir
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        out = self.fc3(x)
        return out

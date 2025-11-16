import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePointMLP(nn.Module):
    """Tiny per-point MLP + global max pooling with layer access."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 64)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        self.layer_outputs = {}

    def forward(self, x: torch.Tensor, save_layers: bool = False) -> torch.Tensor:
        # x: (B, N, 3)
        if save_layers:
            self.layer_outputs['input'] = x.detach().cpu()

        x = F.relu(self.fc1(x))
        if save_layers:
            self.layer_outputs['fc1'] = x.detach().cpu()

        x = F.relu(self.fc2(x))
        if save_layers:
            self.layer_outputs['fc2'] = x.detach().cpu()

        x = F.relu(self.fc3(x))
        if save_layers:
            self.layer_outputs['fc3'] = x.detach().cpu()

        # Global max pooling across points
        x = x.max(dim=1)[0]  # (B, 64)
        if save_layers:
            self.layer_outputs['pooled'] = x.detach().cpu()

        out = self.classifier(x)
        if save_layers:
            self.layer_outputs['output'] = out.detach().cpu()

        return out


class SimplePointCNN(nn.Module):
    """Tiny PointNet-like CNN using 1D convolutions with layer access."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        self.layer_outputs = {}

    def forward(self, x: torch.Tensor, save_layers: bool = False) -> torch.Tensor:
        # x: (B, N, 3)
        if save_layers:
            self.layer_outputs['input'] = x.detach().cpu()

        x = x.transpose(1, 2)  # (B, 3, N)

        x = F.relu(self.conv1(x))
        if save_layers:
            self.layer_outputs['conv1'] = x.transpose(1, 2).detach().cpu()

        x = F.relu(self.conv2(x))
        if save_layers:
            self.layer_outputs['conv2'] = x.transpose(1, 2).detach().cpu()

        x = F.relu(self.conv3(x))
        if save_layers:
            self.layer_outputs['conv3'] = x.transpose(1, 2).detach().cpu()

        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (B, 64)
        if save_layers:
            self.layer_outputs['pooled'] = x.detach().cpu()

        out = self.fc(x)
        if save_layers:
            self.layer_outputs['output'] = out.detach().cpu()

        return out



"""
MLP model definition and training utilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Optional
import numpy as np
from .utils import ModelConfig


class BaseFeatureModel(nn.Module, ABC):
    """
    Base class for models used in this repo's topology/graph detector pipeline.

    Contract:
    - `forward(x)` returns logits of shape (batch, num_classes)
    - `extract_features(x, layer='penultimate')` returns a 2D embedding (batch, feat_dim)

    Any model implementing this contract can be used by:
    - `extract_features_batch(...)`
    - `src.graph_scoring.compute_graph_scores(..., space='feature')`
    - topology detector pipeline in `src.detectors.py`
    """

    @abstractmethod
    def extract_features(self, x: torch.Tensor, layer: str = "penultimate") -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


def get_submodule_by_name(model: nn.Module, dotted_name: str) -> nn.Module:
    """
    Resolve a dotted module path like 'layer4.1.conv2' or 'features.3'.

    This is useful for hooking arbitrary models to expose `extract_features`.
    """
    cur: nn.Module = model
    for part in str(dotted_name).split("."):
        if part.isdigit():
            cur = cur[int(part)]  # type: ignore[index]
        else:
            cur = getattr(cur, part)
    return cur


class HookedFeatureModel(BaseFeatureModel):
    """
    Wrap an arbitrary `nn.Module` and expose `extract_features()` via a forward hook.

    Typical usage (e.g. torchvision ResNet):
        base = resnet18(num_classes=10)
        model = HookedFeatureModel(base, feature_module=get_submodule_by_name(base, "avgpool"))

    The hook output is flattened to (batch, feat_dim) if needed.
    """

    def __init__(self, base: nn.Module, feature_module: nn.Module):
        super().__init__()
        self.base = base
        self._feat: Optional[torch.Tensor] = None

        def _hook(_m, _inp, out):
            # Keep the raw tensor; flattening happens in extract_features.
            if isinstance(out, (tuple, list)):
                out = out[0]
            self._feat = out

        self._hook_handle = feature_module.register_forward_hook(_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)

    def extract_features(self, x: torch.Tensor, layer: str = "penultimate") -> torch.Tensor:
        if layer != "penultimate":
            raise ValueError(f"Unknown layer: {layer}")
        _ = self.base(x)
        if self._feat is None:
            raise RuntimeError("Feature hook did not fire; check feature_module selection.")
        z = self._feat
        if z.ndim > 2:
            z = torch.flatten(z, start_dim=1)
        return z


class TwoMoonsMLP(BaseFeatureModel):
    """Simple MLP classifier for two moons dataset."""
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Optional[list[int]] = None,
        output_dim: int = 2,
        activation: str = 'relu'
    ):
        """
        Initialize MLP model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (number of classes)
            activation: Activation function ('relu' or 'tanh')
        """
        super(TwoMoonsMLP, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.layers(x)
    
    def extract_features(self, x: torch.Tensor, layer: str = 'penultimate') -> torch.Tensor:
        """
        Extract features from a specific layer.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            layer: Which layer to extract from ('penultimate' or 'first')
            
        Returns:
            Feature tensor
        """
        if layer == 'penultimate':
            # Extract features from the last hidden layer
            out = x
            for i, module in enumerate(self.layers):
                out = module(out)
                if i == len(self.layers) - 2:  # Second to last layer
                    return out
            return out
        elif layer == 'first':
            # Extract features from the first hidden layer
            out = x
            for i, module in enumerate(self.layers):
                out = module(out)
                if i == 1:  # After first linear + activation
                    return out
            return out
        else:
            raise ValueError(f"Unknown layer: {layer}")


class CNN(BaseFeatureModel):
    """
    Small CNN used by the synthetic image notebooks (05/06).

    Important: exposes `extract_features(x, layer='penultimate')` so it works with
    the repo's shared helpers (e.g., `extract_features_batch`).
    """

    def __init__(self, num_classes: int = 2, feat_dim: int = 128, in_channels: int = 3):
        super().__init__()
        self.feat_dim = int(feat_dim)
        in_channels = int(in_channels)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, self.feat_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(self.feat_dim, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.extract_features(x, layer="penultimate")
        return self.classifier(z)

    def extract_features(self, x: torch.Tensor, layer: str = "penultimate") -> torch.Tensor:
        if layer != "penultimate":
            raise ValueError(f"Unknown layer: {layer}")
        return self.proj(self.features(x))


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ModelConfig,
    device: str = 'cpu',
    verbose: bool = True
) -> Dict[str, list]:
    """
    Train the model with early stopping based on validation loss.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        
        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, device=device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config.epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    return history


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cpu'
) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def extract_features_batch(
    model: nn.Module,
    X: np.ndarray,
    layer: str = 'penultimate',
    device: str = 'cpu',
    batch_size: int = 32
) -> np.ndarray:
    model.eval()
    model.to(device)
    
    X_tensor = torch.FloatTensor(X).to(device)
    features = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            # Note: model is expected to implement extract_features (TwoMoonsMLP, CNN, etc.)
            batch_features = model.extract_features(batch_X, layer=layer)  # type: ignore[attr-defined]
            features.append(batch_features.cpu().numpy())
    
    return np.vstack(features)


def get_model_predictions(
    model: nn.Module,
    X: np.ndarray,
    device: str = 'cpu',
    return_probs: bool = False
) -> np.ndarray:
    model.eval()
    model = model.to(device)
    
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        if return_probs:
            probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()
        else:
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()


def get_model_logits(
    model: nn.Module,
    X: np.ndarray,
    device: str = 'cpu'
) -> np.ndarray:
    """This is important for our pipeline as it extract logit embeddings from the model"""
    model.eval()
    model = model.to(device)
    
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        return outputs.cpu().numpy()



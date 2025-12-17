"""
MLP model definition and training utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Optional
import numpy as np
from .utils import ModelConfig


class TwoMoonsMLP(nn.Module):
    """Simple MLP classifier for two moons dataset."""
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list = None,
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
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: ModelConfig with training hyperparameters
        device: Device to train on ('cpu' or 'cuda')
        verbose: Whether to print training progress
        
    Returns:
        Dictionary with training history containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'
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
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to evaluate on
        
    Returns:
        Tuple of (average_loss, accuracy_percentage)
    """
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
    """
    Extract features for a batch of inputs.
    
    Args:
        model: Trained PyTorch model
        X: Input array of shape (n_samples, input_dim)
        layer: Which layer to extract from
        device: Device to run on
        batch_size: Batch size for processing
        
    Returns:
        Feature array of shape (n_samples, feature_dim)
    """
    model.eval()
    model = model.to(device)
    
    X_tensor = torch.FloatTensor(X).to(device)
    features = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            batch_features = model.extract_features(batch_X, layer=layer)
            features.append(batch_features.cpu().numpy())
    
    return np.vstack(features)


def get_model_predictions(
    model: nn.Module,
    X: np.ndarray,
    device: str = 'cpu',
    return_probs: bool = False
) -> np.ndarray:
    """
    Get model predictions (or probabilities) for inputs.
    
    Args:
        model: Trained PyTorch model
        X: Input array of shape (n_samples, input_dim)
        device: Device to run on
        return_probs: If True, return probabilities; if False, return class predictions
        
    Returns:
        Array of predictions or probabilities
    """
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
    """
    Get raw logits from model.
    
    Args:
        model: Trained PyTorch model
        X: Input array
        device: Device to run on
        
    Returns:
        Array of logits
    """
    model.eval()
    model = model.to(device)
    
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        return outputs.cpu().numpy()



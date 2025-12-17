"""
Adversarial attack implementations: FGSM and PGD.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from .utils import AttackConfig


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    loss_fn: Optional[nn.Module] = None,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) attack.
    
    Args:
        model: PyTorch model
        x: Input tensor of shape (batch_size, n_features)
        y: True labels tensor of shape (batch_size,)
        epsilon: Attack strength (perturbation bound)
        loss_fn: Loss function (default: CrossEntropyLoss)
        device: Device to run on
        
    Returns:
        Adversarial examples tensor of same shape as x
    """
    model.eval()
    model = model.to(device)
    x = x.to(device).requires_grad_(True)
    y = y.to(device)
    
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    
    # Forward pass
    outputs = model(x)
    loss = loss_fn(outputs, y)
    
    # Backward pass to get gradients
    model.zero_grad()
    loss.backward()
    
    # Create adversarial examples
    x_adv = x + epsilon * x.grad.sign()
    
    return x_adv.detach()


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    num_steps: int = 10,
    step_size: float = 0.01,
    random_start: bool = True,
    loss_fn: Optional[nn.Module] = None,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) attack.
    
    Args:
        model: PyTorch model
        x: Input tensor of shape (batch_size, n_features)
        y: True labels tensor of shape (batch_size,)
        epsilon: Attack strength (perturbation bound)
        num_steps: Number of PGD steps
        step_size: Step size for each iteration
        random_start: Whether to start with random perturbation
        loss_fn: Loss function (default: CrossEntropyLoss)
        device: Device to run on
        
    Returns:
        Adversarial examples tensor of same shape as x
    """
    model.eval()
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)
    
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    
    # Initialize adversarial examples
    if random_start:
        x_adv = x + torch.empty_like(x).uniform_(-epsilon, epsilon)
    else:
        x_adv = x.clone()
    
    # PGD iterations
    for _ in range(num_steps):
        x_adv = x_adv.requires_grad_(True)
        
        # Forward pass
        outputs = model(x_adv)
        loss = loss_fn(outputs, y)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update adversarial examples
        with torch.no_grad():
            grad = x_adv.grad.sign()
            x_adv = x_adv + step_size * grad
            
            # Project back to epsilon-ball around original input
            delta = x_adv - x
            delta = torch.clamp(delta, -epsilon, epsilon)
            x_adv = x + delta
    
    return x_adv.detach()


def generate_adversarial_examples(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    config: AttackConfig,
    device: str = 'cpu',
    batch_size: int = 32
) -> np.ndarray:
    """
    Generate adversarial examples for a dataset using specified attack.
    
    Args:
        model: Trained PyTorch model
        X: Input array of shape (n_samples, n_features)
        y: True labels array of shape (n_samples,)
        config: AttackConfig with attack parameters
        device: Device to run on
        batch_size: Batch size for processing
        
    Returns:
        Adversarial examples array of shape (n_samples, n_features)
    """
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)
    
    X_adv_list = []
    
    for i in range(0, len(X_tensor), batch_size):
        batch_X = X_tensor[i:i+batch_size]
        batch_y = y_tensor[i:i+batch_size]
        
        if config.attack_type == 'fgsm':
            batch_X_adv = fgsm_attack(
                model, batch_X, batch_y, config.epsilon, device=device
            )
        elif config.attack_type == 'pgd':
            batch_X_adv = pgd_attack(
                model, batch_X, batch_y,
                epsilon=config.epsilon,
                num_steps=config.num_steps,
                step_size=config.step_size,
                random_start=config.random_start,
                device=device
            )
        else:
            raise ValueError(f"Unknown attack type: {config.attack_type}")
        
        X_adv_list.append(batch_X_adv.cpu().numpy())
    
    return np.vstack(X_adv_list)


def compute_attack_success_rate(
    model: nn.Module,
    X_original: np.ndarray,
    X_adv: np.ndarray,
    y_true: np.ndarray,
    device: str = 'cpu'
) -> float:
    """
    Compute the attack success rate (percentage of adversarial examples that fool the model).
    
    Args:
        model: Trained PyTorch model
        X_original: Original inputs
        X_adv: Adversarial inputs
        y_true: True labels
        device: Device to run on
        
    Returns:
        Attack success rate (percentage)
    """
    from .models import get_model_predictions
    
    # Predictions on original examples
    pred_original = get_model_predictions(model, X_original, device=device)
    
    # Predictions on adversarial examples
    pred_adv = get_model_predictions(model, X_adv, device=device)
    
    # Count where original was correct but adversarial is wrong
    original_correct = (pred_original == y_true)
    adversarial_wrong = (pred_adv != y_true)
    successful_attacks = original_correct & adversarial_wrong
    
    if original_correct.sum() == 0:
        return 0.0
    
    success_rate = 100.0 * successful_attacks.sum() / original_correct.sum()
    return success_rate


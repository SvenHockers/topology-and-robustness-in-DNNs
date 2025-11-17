from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Callable
import torch
import torch.nn.functional as F


@dataclass
class AttackResult:
    """Result from an adversarial attack."""
    x_adv: torch.Tensor
    success: bool  # Whether the attack succeeded (prediction flipped)
    metadata: Dict[str, Any] = None  # Additional attack-specific information
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# Projection functions (reused from probes.py for consistency)
def _linf_project(x_adv: torch.Tensor, x_orig: torch.Tensor, eps: float) -> torch.Tensor:
    """Project x_adv into L∞ ball of radius eps around x_orig."""
    eta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
    return x_orig + eta


def _l2_project(x_adv: torch.Tensor, x_orig: torch.Tensor, eps: float) -> torch.Tensor:
    """Project x_adv into L2 ball of radius eps around x_orig."""
    delta = x_adv - x_orig
    norm = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1, keepdim=True) + 1e-12
    factor = torch.clamp(eps / norm, max=1.0).view(-1, 1, 1)
    return x_orig + delta * factor


# Attack registry for extensibility
_ATTACK_REGISTRY: Dict[str, Callable] = {}


def register_attack(name: str):
    """Decorator to register an attack function."""
    def decorator(func: Callable):
        _ATTACK_REGISTRY[name] = func
        return func
    return decorator


def get_attack(name: str) -> Optional[Callable]:
    """Get an attack function by name."""
    return _ATTACK_REGISTRY.get(name)


def list_attacks() -> list[str]:
    """List all registered attack names."""
    return list(_ATTACK_REGISTRY.keys())


# ============================================================================
# FGSM (Fast Gradient Sign Method)
# ============================================================================

@register_attack("fgsm")
def fgsm_attack(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    norm: str,
    eps: float,
) -> AttackResult:
    """
    Fast Gradient Sign Method - single-step attack.
    
    Args:
        model: PyTorch model
        x: Input tensor (any shape, typically (1, N, D) for point clouds)
        y: True label (scalar tensor)
        norm: "linf" or "l2"
        eps: Perturbation budget
        
    Returns:
        AttackResult with adversarial example
    """
    model.eval()
    x_orig = x.detach()
    x_adv = x_orig.clone()
    x_adv.requires_grad_(True)
    
    # Forward pass
    logits = model(x_adv, save_layers=False)
    loss = F.cross_entropy(logits, y.unsqueeze(0) if y.dim() == 0 else y)
    
    # Backward pass
    model.zero_grad()
    if x_adv.grad is not None:
        x_adv.grad.zero_()
    loss.backward()
    grad = x_adv.grad
    
    # Apply perturbation based on norm
    if norm == "linf":
        # FGSM: x_adv = x + eps * sign(gradient)
        x_adv = x_orig + eps * grad.sign()
        x_adv = _linf_project(x_adv, x_orig, eps)
    elif norm == "l2":
        # L2 version: x_adv = x + eps * (gradient / ||gradient||)
        grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1, keepdim=True) + 1e-12
        grad_unit = (grad / grad_norm).view_as(grad)
        x_adv = x_orig + eps * grad_unit
        x_adv = _l2_project(x_adv, x_orig, eps)
    else:
        raise ValueError(f"Unsupported norm for FGSM: {norm}")
    
    x_adv = x_adv.detach()
    
    # Check if attack succeeded
    with torch.no_grad():
        pred_adv = model(x_adv, save_layers=False).argmax(1).item()
        pred_orig = model(x_orig, save_layers=False).argmax(1).item()
        success = pred_adv != pred_orig
    
    # Compute perturbation magnitude
    delta = x_adv - x_orig
    if norm == "linf":
        pert_mag = float(delta.abs().max().item())
    else:  # l2
        pert_mag = float(torch.norm(delta.view(delta.size(0), -1), p=2, dim=1).item())
    
    metadata = {
        "perturbation_magnitude": pert_mag,
        "original_prediction": pred_orig,
        "adversarial_prediction": pred_adv,
    }
    
    return AttackResult(x_adv=x_adv, success=success, metadata=metadata)


def estimate_min_eps_fgsm(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    norm: str,
    eps_max: float,
    tol: float,
    max_outer: int = 12,
) -> Tuple[Optional[float], Optional[torch.Tensor]]:
    """
    Bisection search for minimal epsilon using FGSM as inner oracle.
    Returns (eps_star, x_adv_at_eps_star).
    """
    device = next(model.parameters()).device
    # Ensure x has batch dimension: (N, 3) -> (1, N, 3) or keep (1, N, 3) as is
    if x.dim() == 2:
        x = x.unsqueeze(0)  # Add batch dimension
    x = x.to(device)
    y = y.to(device)
    
    with torch.no_grad():
        clean_pred = model(x, save_layers=False).argmax(1).item()
    if clean_pred != int(y.item()):
        return 0.0, x.detach().cpu()
    
    lo, hi = 0.0, eps_max
    found = None
    x_best = None
    for _ in range(max_outer):
        mid = 0.5 * (lo + hi)
        result = fgsm_attack(model, x.clone(), y, norm=norm, eps=mid)
        with torch.no_grad():
            pred = model(result.x_adv, save_layers=False).argmax(1).item()
        if pred != int(y.item()):
            found = mid
            x_best = result.x_adv
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return found, (x_best.detach().cpu() if x_best is not None else None)


# ============================================================================
# PGD (Projected Gradient Descent) - moved from probes.py for consistency
# ============================================================================

@register_attack("pgd")
def pgd_attack(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    norm: str,
    eps: float,
    steps: int,
    step_frac: float = 1.0,
    random_start: bool = True,
) -> AttackResult:
    """
    PGD attack in L_inf or L2.
    x: (1, N, 3) or any tensor shape, y: scalar tensor
    """
    model.eval()
    x_orig = x.detach()
    x_adv = x_orig.clone()
    if random_start:
        if norm == "linf":
            x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
        elif norm == "l2":
            noise = torch.randn_like(x_orig)
            noise = noise / (torch.norm(noise.view(1, -1), p=2, dim=1, keepdim=True) + 1e-12).view(1, 1, 1)
            x_adv = x_orig + eps * noise
    x_adv.requires_grad_(True)

    alpha = step_frac * (eps / max(steps, 1))
    for _ in range(steps):
        logits = model(x_adv, save_layers=False)
        loss = F.cross_entropy(logits, y.unsqueeze(0) if y.dim() == 0 else y)
        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss.backward()
        grad = x_adv.grad
        if norm == "linf":
            x_adv = x_adv + alpha * grad.sign()
            x_adv = _linf_project(x_adv, x_orig, eps)
        elif norm == "l2":
            grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1, keepdim=True) + 1e-12
            x_adv = x_adv + alpha * (grad / grad_norm).view_as(x_adv)
            x_adv = _l2_project(x_adv, x_orig, eps)
        else:
            raise ValueError(f"Unsupported norm: {norm}")
        x_adv = x_adv.detach().requires_grad_(True)
    
    x_adv = x_adv.detach()
    
    # Check if attack succeeded
    with torch.no_grad():
        pred_adv = model(x_adv, save_layers=False).argmax(1).item()
        pred_orig = model(x_orig, save_layers=False).argmax(1).item()
        success = pred_adv != pred_orig
    
    # Compute perturbation magnitude
    delta = x_adv - x_orig
    if norm == "linf":
        pert_mag = float(delta.abs().max().item())
    else:  # l2
        pert_mag = float(torch.norm(delta.view(delta.size(0), -1), p=2, dim=1).item())
    
    metadata = {
        "perturbation_magnitude": pert_mag,
        "original_prediction": pred_orig,
        "adversarial_prediction": pred_adv,
        "steps": steps,
    }
    
    return AttackResult(x_adv=x_adv, success=success, metadata=metadata)


# ============================================================================
# L0 Sparse Attack
# ============================================================================

@register_attack("l0")
def l0_attack(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    max_perturbed_elements: int,
    strategy: str = "gradient_based",
    max_iterations: int = 100,
    eps_per_element: float = 0.1,
) -> AttackResult:
    """
    L0 sparse attack - modifies only a few elements.
    
    Args:
        model: PyTorch model
        x: Input tensor (any shape)
        y: True label (scalar tensor)
        max_perturbed_elements: Maximum number of elements to modify
        strategy: "gradient_based", "random", or "furthest"
        max_iterations: Maximum optimization iterations
        eps_per_element: Perturbation magnitude per modified element
        
    Returns:
        AttackResult with sparse adversarial example
    """
    model.eval()
    x_orig = x.detach()
    x_adv = x_orig.clone()
    
    # Flatten for element selection, then reshape back
    original_shape = x_adv.shape
    x_flat = x_adv.view(-1)
    n_elements = x_flat.numel()
    
    # Select elements to perturb
    if strategy == "gradient_based":
        x_adv.requires_grad_(True)
        logits = model(x_adv, save_layers=False)
        loss = F.cross_entropy(logits, y.unsqueeze(0) if y.dim() == 0 else y)
        model.zero_grad()
        loss.backward()
        grad_flat = x_adv.grad.view(-1).abs()
        # Select top-k elements by gradient magnitude
        _, top_indices = torch.topk(grad_flat, min(max_perturbed_elements, n_elements))
    elif strategy == "random":
        top_indices = torch.randperm(n_elements)[:min(max_perturbed_elements, n_elements)]
    elif strategy == "furthest":
        # Select elements furthest from mean (for point clouds, this might be outliers)
        mean_val = x_flat.mean()
        distances = (x_flat - mean_val).abs()
        _, top_indices = torch.topk(distances, min(max_perturbed_elements, n_elements))
    else:
        raise ValueError(f"Unknown L0 strategy: {strategy}")
    
    # Iteratively optimize perturbation on selected elements
    x_adv = x_orig.clone()
    for iteration in range(max_iterations):
        x_adv.requires_grad_(True)
        logits = model(x_adv, save_layers=False)
        loss = F.cross_entropy(logits, y.unsqueeze(0) if y.dim() == 0 else y)
        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss.backward()
        
        # Get gradient and detach before modifying
        grad_flat = x_adv.grad.view(-1).detach()
        x_orig_flat = x_orig.view(-1)
        
        # Create new tensor for x_adv (don't modify in-place)
        x_flat_new = x_adv.detach().view(-1).clone()
        
        # Update only selected indices
        for idx in top_indices:
            grad_val = grad_flat[idx]
            # Perturb in direction of gradient
            x_flat_new[idx] = x_orig_flat[idx] + eps_per_element * grad_val.sign()
        
        x_adv = x_flat_new.view(original_shape)
        
        # Check if attack succeeded
        with torch.no_grad():
            pred_adv = model(x_adv, save_layers=False).argmax(1).item()
            pred_orig = model(x_orig, save_layers=False).argmax(1).item()
            if pred_adv != pred_orig:
                break
    
    x_adv = x_adv.detach()
    
    # Check final success
    with torch.no_grad():
        pred_adv = model(x_adv, save_layers=False).argmax(1).item()
        pred_orig = model(x_orig, save_layers=False).argmax(1).item()
        success = pred_adv != pred_orig
    
    # Count actual modified elements
    delta = (x_adv - x_orig).abs()
    n_modified = (delta > 1e-6).sum().item()
    
    # Compute L0 and L2 perturbation magnitudes
    delta_flat = delta.view(-1)
    l0_mag = float((delta_flat > 1e-6).sum().item())
    l2_mag = float(torch.norm(delta_flat, p=2).item())
    linf_mag = float(delta_flat.max().item())
    
    metadata = {
        "n_modified_elements": n_modified,
        "l0_magnitude": l0_mag,
        "l2_magnitude": l2_mag,
        "linf_magnitude": linf_mag,
        "original_prediction": pred_orig,
        "adversarial_prediction": pred_adv,
        "iterations": iteration + 1,
    }
    
    return AttackResult(x_adv=x_adv, success=success, metadata=metadata)


# ============================================================================
# Carlini-Wagner (CW) Attack
# ============================================================================

@register_attack("cw")
def cw_attack(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    norm: str,
    c_init: float = 0.001,
    c_max: float = 10.0,
    binary_search_steps: int = 9,
    max_iterations: int = 1000,
    learning_rate: float = 0.01,
    confidence: float = 0.0,
) -> AttackResult:
    """
    Carlini-Wagner attack - optimization-based attack that minimizes perturbation.
    
    Args:
        model: PyTorch model
        x: Input tensor
        y: True label
        norm: "linf" or "l2"
        c_init: Initial value for c parameter
        c_max: Maximum value for c parameter
        binary_search_steps: Number of binary search steps for c
        max_iterations: Maximum optimization iterations per c value
        learning_rate: Learning rate for optimization
        confidence: Confidence margin (kappa in CW paper)
        
    Returns:
        AttackResult with minimal perturbation adversarial example
    """
    model.eval()
    device = next(model.parameters()).device
    x_orig = x.detach().to(device)
    y = y.to(device)
    
    # Simplified CW: optimize directly without tanh parameterization
    # For full implementation, tanh parameterization is better but more complex
    best_x_adv = None
    best_pert_mag = float('inf')
    c_low, c_high = 0.0, c_max
    
    for bs_step in range(binary_search_steps):
        c = (c_low + c_high) / 2.0 if bs_step > 0 else c_init
        
        x_adv = x_orig.clone()
        x_adv.requires_grad_(True)
        optimizer = torch.optim.Adam([x_adv], lr=learning_rate)
        
        for iteration in range(max_iterations):
            logits = model(x_adv, save_layers=False)
            
            # Perturbation loss
            delta = x_adv - x_orig
            if norm == "linf":
                pert_loss = delta.abs().max()
            else:  # l2
                pert_loss = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1).mean()
            
            # CW objective: f(x) = max(0, max_{i≠t} logits[i] - logits[t] + confidence)
            true_logit = logits[0, int(y.item())]
            other_logits = torch.cat([logits[0, :int(y.item())], logits[0, int(y.item())+1:]])
            max_other_logit = other_logits.max()
            f_loss = torch.clamp(max_other_logit - true_logit + confidence, min=0.0)
            
            total_loss = pert_loss + c * f_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Project to reasonable bounds (adaptive)
            if norm == "linf":
                # Use adaptive eps based on current perturbation
                current_eps = delta.abs().max().item()
                if current_eps > 1.0:  # Reasonable bound
                    x_adv.data = _linf_project(x_adv.data, x_orig, 1.0)
            # For L2, we let it optimize freely
        
        # Check result
        with torch.no_grad():
            delta = x_adv - x_orig
            if norm == "linf":
                pert_mag = float(delta.abs().max().item())
            else:
                pert_mag = float(torch.norm(delta.view(delta.size(0), -1), p=2, dim=1).item())
            
            pred_adv = model(x_adv, save_layers=False).argmax(1).item()
            pred_orig = model(x_orig, save_layers=False).argmax(1).item()
            success = pred_adv != int(y.item())
        
        # Binary search update
        if success and pert_mag < best_pert_mag:
            best_x_adv = x_adv.detach().clone()
            best_pert_mag = pert_mag
            c_high = c
        else:
            c_low = c
        
        if c_high - c_low < 1e-5:
            break
    
    if best_x_adv is None:
        best_x_adv = x_adv.detach()
        best_pert_mag = pert_mag
    
    # Final check
    with torch.no_grad():
        pred_adv = model(best_x_adv, save_layers=False).argmax(1).item()
        pred_orig = model(x_orig, save_layers=False).argmax(1).item()
        success = pred_adv != int(y.item())
    
    metadata = {
        "perturbation_magnitude": best_pert_mag,
        "original_prediction": pred_orig,
        "adversarial_prediction": pred_adv,
        "c_final": (c_low + c_high) / 2.0,
    }
    
    return AttackResult(x_adv=best_x_adv.cpu(), success=success, metadata=metadata)


# ============================================================================
# Universal Adversarial Perturbations (UAP)
# ============================================================================

@register_attack("uap")
def uap_attack(
    model,
    x_samples: torch.Tensor,  # Batch of samples
    y_samples: torch.Tensor,  # Batch of labels
    max_iterations: int = 1000,
    delta_init: float = 0.01,
    xi: float = 10.0,
    eps: float = 0.1,
    norm: str = "linf",
) -> AttackResult:
    """
    Universal Adversarial Perturbations - single perturbation that fools multiple samples.
    
    Args:
        model: PyTorch model
        x_samples: Batch of input tensors (B, N, D) or (B, ...)
        y_samples: Batch of true labels (B,)
        max_iterations: Maximum optimization iterations
        delta_init: Initial perturbation magnitude
        xi: Fooling rate threshold (percentage)
        eps: Maximum perturbation budget
        norm: "linf" or "l2"
        
    Returns:
        AttackResult with universal perturbation
    """
    model.eval()
    device = next(model.parameters()).device
    x_samples = x_samples.to(device)
    y_samples = y_samples.to(device)
    batch_size = x_samples.size(0)
    
    # Initialize universal perturbation
    v = torch.zeros_like(x_samples[0:1], requires_grad=True)
    optimizer = torch.optim.Adam([v], lr=delta_init)
    
    fooled_count = 0
    for iteration in range(max_iterations):
        fooled_this_iter = 0
        total_loss = 0.0
        
        for i in range(batch_size):
            x_i = x_samples[i:i+1]
            y_i = y_samples[i:i+1]
            x_pert = x_i + v
            
            # Project to constraint
            if norm == "linf":
                x_pert = _linf_project(x_pert, x_i, eps)
            else:  # l2
                x_pert = _l2_project(x_pert, x_i, eps)
            
            logits = model(x_pert, save_layers=False)
            loss = F.cross_entropy(logits, y_i)
            total_loss += loss
            
            # Check if fooled
            with torch.no_grad():
                pred = logits.argmax(1).item()
                if pred != int(y_i.item()):
                    fooled_this_iter += 1
        
        # Optimize
        optimizer.zero_grad()
        (total_loss / batch_size).backward()
        optimizer.step()
        
        # Project v to constraint
        if norm == "linf":
            v.data = torch.clamp(v.data, min=-eps, max=eps)
        else:  # l2
            v_norm = torch.norm(v.data.view(1, -1), p=2, dim=1) + 1e-12
            if v_norm > eps:
                v.data = v.data * (eps / v_norm).view(-1, 1, 1)
        
        fooled_count = fooled_this_iter
        fooling_rate = 100.0 * fooled_count / batch_size
        
        if fooling_rate >= xi:
            break
    
    v_final = v.detach()
    
    # Evaluate final fooling rate
    fooled_final = 0
    for i in range(batch_size):
        x_i = x_samples[i:i+1]
        y_i = y_samples[i:i+1]
        x_pert = x_i + v_final
        
        if norm == "linf":
            x_pert = _linf_project(x_pert, x_i, eps)
        else:
            x_pert = _l2_project(x_pert, x_i, eps)
        
        with torch.no_grad():
            logits = model(x_pert, save_layers=False)
            pred = logits.argmax(1).item()
            if pred != int(y_i.item()):
                fooled_final += 1
    
    fooling_rate_final = 100.0 * fooled_final / batch_size
    success = fooling_rate_final >= xi
    
    # Measure perturbation magnitude
    if norm == "linf":
        pert_mag = float(v_final.abs().max().item())
    else:
        pert_mag = float(torch.norm(v_final.view(1, -1), p=2, dim=1).item())
    
    metadata = {
        "perturbation_magnitude": pert_mag,
        "fooling_rate": fooling_rate_final,
        "n_fooled": fooled_final,
        "n_samples": batch_size,
        "iterations": iteration + 1,
    }
    
    # Return first sample with UAP applied as representative
    x_adv_rep = (x_samples[0:1] + v_final).cpu()
    if norm == "linf":
        x_adv_rep = _linf_project(x_adv_rep, x_samples[0:1].cpu(), eps)
    else:
        x_adv_rep = _l2_project(x_adv_rep, x_samples[0:1].cpu(), eps)
    
    return AttackResult(x_adv=x_adv_rep, success=success, metadata=metadata)


# ============================================================================
# Boundary Attack
# ============================================================================

@register_attack("boundary")
def boundary_attack(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    max_iterations: int = 1000,
    spherical_step_size: float = 0.01,
    source_step_size: float = 0.01,
    init_perturbation: Optional[torch.Tensor] = None,
) -> AttackResult:
    """
    Boundary Attack - decision-based attack that reduces perturbation while staying adversarial.
    
    Args:
        model: PyTorch model
        x: Input tensor
        y: True label
        max_iterations: Maximum iterations
        spherical_step_size: Step size for spherical step
        source_step_size: Step size for source step
        init_perturbation: Initial adversarial example (if None, uses random large perturbation)
        
    Returns:
        AttackResult with minimal perturbation adversarial example
    """
    model.eval()
    device = next(model.parameters()).device
    x_orig = x.detach().to(device)
    y = y.to(device)
    
    # Check if already misclassified
    with torch.no_grad():
        pred_orig = model(x_orig, save_layers=False).argmax(1).item()
        if pred_orig != int(y.item()):
            return AttackResult(
                x_adv=x_orig.cpu(),
                success=True,
                metadata={"perturbation_magnitude": 0.0, "iterations": 0}
            )
    
    # Initialize: find an adversarial example (use large random perturbation)
    if init_perturbation is None:
        # Start with large perturbation
        x_adv = x_orig.clone()
        for _ in range(100):  # Try up to 100 random perturbations
            noise = torch.randn_like(x_orig) * 0.5
            x_candidate = x_orig + noise
            with torch.no_grad():
                pred = model(x_candidate, save_layers=False).argmax(1).item()
                if pred != int(y.item()):
                    x_adv = x_candidate
                    break
    else:
        x_adv = init_perturbation.to(device)
    
    # Verify initial is adversarial
    with torch.no_grad():
        pred_init = model(x_adv, save_layers=False).argmax(1).item()
        if pred_init == int(y.item()):
            # Failed to find initial adversarial example
            return AttackResult(
                x_adv=x_orig.cpu(),
                success=False,
                metadata={"perturbation_magnitude": float('inf'), "iterations": 0}
            )
    
    # Iterative reduction
    for iteration in range(max_iterations):
        # Spherical step: move towards source along sphere
        delta = x_adv - x_orig
        delta_norm = torch.norm(delta.view(1, -1), p=2, dim=1) + 1e-12
        delta_unit = (delta / delta_norm).view_as(delta)
        
        # Move towards source
        x_candidate = x_adv - spherical_step_size * delta_unit
        
        # Source step: small random step orthogonal to delta
        # Generate random direction
        random_dir = torch.randn_like(x_adv)
        # Project to be orthogonal to delta
        random_dir_flat = random_dir.view(1, -1)
        delta_flat = delta.view(1, -1)
        proj = (random_dir_flat @ delta_flat.T) / (delta_norm ** 2 + 1e-12)
        random_dir_flat = random_dir_flat - proj * delta_flat
        random_dir = random_dir_flat.view_as(x_adv)
        random_dir = random_dir / (torch.norm(random_dir.view(1, -1), p=2, dim=1) + 1e-12).view(-1, 1, 1)
        
        x_candidate = x_candidate + source_step_size * random_dir
        
        # Check if still adversarial
        with torch.no_grad():
            pred = model(x_candidate, save_layers=False).argmax(1).item()
            if pred != int(y.item()):
                # Accept move
                x_adv = x_candidate
            # else: reject and continue
        
        # Early stopping if perturbation is very small
        delta_current = x_adv - x_orig
        pert_mag = float(torch.norm(delta_current.view(1, -1), p=2, dim=1).item())
        if pert_mag < 1e-6:
            break
    
    x_adv = x_adv.detach()
    
    # Final check
    with torch.no_grad():
        pred_adv = model(x_adv, save_layers=False).argmax(1).item()
        success = pred_adv != int(y.item())
    
    # Measure perturbation
    delta = x_adv - x_orig
    pert_mag = float(torch.norm(delta.view(1, -1), p=2, dim=1).item())
    pert_mag_linf = float(delta.abs().max().item())
    
    metadata = {
        "perturbation_magnitude": pert_mag,
        "perturbation_magnitude_linf": pert_mag_linf,
        "original_prediction": pred_orig,
        "adversarial_prediction": pred_adv,
        "iterations": iteration + 1,
    }
    
    return AttackResult(x_adv=x_adv.cpu(), success=success, metadata=metadata)


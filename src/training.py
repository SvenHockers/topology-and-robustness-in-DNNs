import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any


def _generate_adversarial_example(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    attack_type: str,
    norm: str,
    pgd_steps: int = 2,
    step_size: Optional[float] = None,
) -> torch.Tensor:
    """
    Generate adversarial example for training.
    
    Args:
        model: PyTorch model (should be in train mode for training-time attacks)
        x: Clean input tensor (B, N, D)
        y: True labels (B,)
        eps: Perturbation budget
        attack_type: "fgsm" or "pgd"
        norm: "linf" or "l2"
        pgd_steps: Number of steps for PGD (default: 2 for fast training)
        step_size: Step size for PGD (if None, auto: eps / pgd_steps)
        
    Returns:
        Adversarial example tensor (B, N, D)
    """
    x_orig = x.detach()
    x_adv = x_orig.clone()
    x_adv.requires_grad_(True)
    
    # Projection functions
    def _linf_project(x_adv, x_orig, eps):
        delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        return x_orig + delta
    
    def _l2_project(x_adv, x_orig, eps):
        delta = x_adv - x_orig
        # Flatten spatial dimensions for norm computation
        delta_flat = delta.view(delta.size(0), -1)
        norm = torch.norm(delta_flat, p=2, dim=1, keepdim=True) + 1e-12
        factor = torch.clamp(eps / norm, max=1.0)
        # Reshape factor to match delta shape
        while factor.dim() < delta.dim():
            factor = factor.unsqueeze(-1)
        return x_orig + delta * factor
    
    project_fn = _linf_project if norm == "linf" else _l2_project
    
    if attack_type == "fgsm":
        # Single-step FGSM
        logits = model(x_adv, save_layers=False)
        loss = F.cross_entropy(logits, y)
        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss.backward()
        
        grad = x_adv.grad
        if norm == "linf":
            x_adv = x_orig + eps * torch.sign(grad)
        else:  # l2
            grad_flat = grad.view(grad.size(0), -1)
            grad_norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True) + 1e-12
            grad_unit = grad / grad_norm.view(-1, 1, 1)
            x_adv = x_orig + eps * grad_unit
        
        x_adv = project_fn(x_adv, x_orig, eps)
        
    elif attack_type == "pgd":
        # Multi-step PGD
        # Random start (small random initialization)
        if norm == "linf":
            x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
        else:  # l2
            noise = torch.randn_like(x_orig)
            noise_flat = noise.view(noise.size(0), -1)
            noise_norm = torch.norm(noise_flat, p=2, dim=1, keepdim=True) + 1e-12
            noise_unit = noise / noise_norm.view(-1, 1, 1)
            x_adv = x_orig + eps * noise_unit * torch.rand(x_orig.size(0), 1, 1, device=x_orig.device)
        
        x_adv = project_fn(x_adv, x_orig, eps)
        x_adv.requires_grad_(True)
        
        # PGD steps: use provided step_size or auto-compute
        if step_size is None:
            alpha = eps / pgd_steps
        else:
            alpha = step_size
        for _ in range(pgd_steps):
            logits = model(x_adv, save_layers=False)
            loss = F.cross_entropy(logits, y)
            model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.zero_()
            loss.backward()
            
            grad = x_adv.grad
            if norm == "linf":
                x_adv = x_adv.detach() + alpha * torch.sign(grad)
            else:  # l2
                grad_flat = grad.view(grad.size(0), -1)
                grad_norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True) + 1e-12
                grad_unit = grad / grad_norm.view(-1, 1, 1)
                x_adv = x_adv.detach() + alpha * grad_unit
            
            x_adv = project_fn(x_adv, x_orig, eps)
            x_adv.requires_grad_(True)
    else:
        raise ValueError(f"Unknown attack_type: {attack_type}")
    
    return x_adv.detach()


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    adv_config: Optional[Dict[str, Any]] = None,
):
    """
    Train one epoch with optional topology-preserving adversarial training.
    
    Extended version with feature consistency (fc3 + pooled) and logit consistency.
    
    Args:
        model: PyTorch model
        loader: DataLoader
        optimizer: Optimizer
        criterion: Loss function (e.g., CrossEntropyLoss)
        device: Device
        adv_config: Optional dict with keys:
            - enabled: bool (default: False)
            - epsilon: float (default: 0.1)
            - adv_steps: int (default: 2) - PGD steps for training
            - adv_step_size: Optional[float] (default: None, auto: epsilon / adv_steps)
            - lambda_adv: float (default: 1.0)
            - lambda_feat_fc3: float (default: 0.1) - fc3 feature consistency weight
            - lambda_feat_pooled: float (default: 0.1) - pooled feature consistency weight
            - lambda_logit: float (default: 0.1) - logit consistency (KL) weight
            - attack_type: str (default: "pgd")
            - norm: str (default: "linf")
            # Legacy fields (for backwards compatibility):
            - pgd_steps: int (deprecated, use adv_steps)
            - lambda_rep: float (deprecated, use lambda_feat_pooled)
            - rep_layer: str (deprecated, now always uses both fc3 and pooled)
    
    Returns:
        Tuple of (avg_loss, accuracy) if adv_config disabled, or dict with detailed stats if enabled
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    # Adversarial training stats
    clean_loss_sum = 0.0
    adv_loss_sum = 0.0
    feat_fc3_loss_sum = 0.0
    feat_pooled_loss_sum = 0.0
    logit_loss_sum = 0.0
    clean_correct, adv_correct = 0, 0
    
    # Parse adversarial config (backwards compatible: if None or disabled, use standard training)
    use_adv = False
    if adv_config is not None and adv_config.get("enabled", False):
        use_adv = True
        eps = adv_config.get("epsilon", 0.1)
        # New fields (preferred)
        adv_steps = adv_config.get("adv_steps", adv_config.get("pgd_steps", 2))  # Support legacy pgd_steps
        adv_step_size = adv_config.get("adv_step_size", None)
        lambda_adv = adv_config.get("lambda_adv", 1.0)
        lambda_feat_fc3 = adv_config.get("lambda_feat_fc3", adv_config.get("lambda_rep", 0.1))  # Legacy fallback
        lambda_feat_pooled = adv_config.get("lambda_feat_pooled", adv_config.get("lambda_rep", 0.1))  # Legacy fallback
        lambda_logit = adv_config.get("lambda_logit", 0.1)
        attack_type = adv_config.get("attack_type", "pgd")
        norm = adv_config.get("norm", "linf")
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        if use_adv:
            # ============================================================
            # TOPOLOGY-PRESERVING ADVERSARIAL TRAINING
            # Extended version with feature + logit consistency
            # ============================================================
            
            # 1. Clean forward pass (with layer saving for feature consistency)
            logits_clean = model(x, save_layers=True)
            loss_clean = criterion(logits_clean, y)
            
            # Extract clean features: fc3 (before pooling) and pooled (after pooling)
            if 'fc3' not in model.layer_outputs or 'pooled' not in model.layer_outputs:
                raise ValueError(f"Required layers 'fc3' and 'pooled' not found in model.layer_outputs. "
                               f"Available layers: {list(model.layer_outputs.keys())}")
            
            fc3_clean = model.layer_outputs['fc3'].to(device)  # (B, N, 64)
            pooled_clean = model.layer_outputs['pooled'].to(device)  # (B, 64)
            
            # 2. Generate adversarial example using PGD with configurable steps
            x_adv = _generate_adversarial_example(
                model, x, y, eps, attack_type, norm, adv_steps, adv_step_size
            )
            
            # 3. Adversarial forward pass (with layer saving)
            logits_adv = model(x_adv, save_layers=True)
            loss_adv = criterion(logits_adv, y)
            
            # 4. Extract adversarial features
            fc3_adv = model.layer_outputs['fc3'].to(device)  # (B, N, 64)
            pooled_adv = model.layer_outputs['pooled'].to(device)  # (B, 64)
            
            # 5. Feature consistency losses (L2/MSE)
            # fc3: before pooling (point-wise features)
            feat_fc3_loss = F.mse_loss(fc3_clean, fc3_adv)
            # pooled: after pooling (global representation)
            feat_pooled_loss = F.mse_loss(pooled_clean, pooled_adv)
            
            # 6. Logit consistency loss (KL divergence: p_clean || p_adv)
            # KL(p_adv || p_clean) = sum(p_adv * log(p_adv / p_clean))
            # Using F.kl_div with log_softmax of adv and softmax of clean
            p_clean = F.softmax(logits_clean, dim=1)  # (B, num_classes)
            log_p_adv = F.log_softmax(logits_adv, dim=1)  # (B, num_classes)
            logit_loss = F.kl_div(log_p_adv, p_clean, reduction='batchmean')
            
            # 7. Combined loss
            loss = (loss_clean 
                   + lambda_adv * loss_adv
                   + lambda_feat_fc3 * feat_fc3_loss
                   + lambda_feat_pooled * feat_pooled_loss
                   + lambda_logit * logit_loss)
            
            # Accumulate stats
            clean_loss_sum += float(loss_clean.item()) * x.size(0)
            adv_loss_sum += float(loss_adv.item()) * x.size(0)
            feat_fc3_loss_sum += float(feat_fc3_loss.item()) * x.size(0)
            feat_pooled_loss_sum += float(feat_pooled_loss.item()) * x.size(0)
            logit_loss_sum += float(logit_loss.item()) * x.size(0)
            
            # Track accuracies
            clean_correct += int((logits_clean.argmax(1) == y).sum().item())
            adv_correct += int((logits_adv.argmax(1) == y).sum().item())
            
        else:
            # ============================================================
            # STANDARD CLEAN-ONLY TRAINING (original behavior)
            # ============================================================
            preds_clean = model(x, save_layers=False)
            loss = criterion(preds_clean, y)
            clean_loss_sum += float(loss.item()) * x.size(0)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += float(loss.item()) * x.size(0)
        if use_adv:
            # For adversarial training, use clean predictions for main accuracy
            correct += clean_correct
        else:
            correct += int((preds_clean.argmax(1) == y).sum().item())
        total += int(x.size(0))
    
    avg_loss = total_loss / max(total, 1)
    
    if use_adv:
        # For adversarial training, use clean accuracy as main accuracy metric
        accuracy = clean_correct / max(total, 1)
        # Return detailed stats for adversarial training
        return {
            "total_loss": avg_loss,
            "accuracy": accuracy,
            "clean_loss": clean_loss_sum / max(total, 1),
            "adv_loss": adv_loss_sum / max(total, 1),
            "feat_fc3_loss": feat_fc3_loss_sum / max(total, 1),
            "feat_pooled_loss": feat_pooled_loss_sum / max(total, 1),
            "logit_loss": logit_loss_sum / max(total, 1),
            "clean_acc": clean_correct / max(total, 1),
            "adv_acc": adv_correct / max(total, 1),
        }
    else:
        # Return simple tuple for backwards compatibility (standard training)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x, save_layers=False)
            loss = criterion(preds, y)
            total_loss += float(loss.item()) * x.size(0)
            correct += int((preds.argmax(1) == y).sum().item())
            total += int(x.size(0))
    return total_loss / max(total, 1), correct / max(total, 1)


def show_some_predictions(model, loader, device, n_show: int = 10):
    """Print a few (true_label, predicted_label) pairs."""
    model.eval()
    shown = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, save_layers=False)
            preds = logits.argmax(1)
            for i in range(x.size(0)):
                print(f"true={int(y[i])}, pred={int(preds[i])}")
                shown += 1
                if shown >= n_show:
                    return



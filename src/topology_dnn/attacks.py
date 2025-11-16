import torch
import torch.nn.functional as F


def find_min_adversarial_perturbation_iterative(
    model, x, y, device, eps_max: float = 1.0, n_steps: int = 40
):
    """
    Iterative PGD-style attack to approximate the smallest L_inf perturbation
    (epsilon) along a gradient-based path that flips the prediction.

    Args:
        model: trained model
        x: tensor (N, 3) point cloud (no batch dim)
        y: tensor scalar (true label)
        device: torch.device
        eps_max: maximum allowed L_inf radius for the whole attack
        n_steps: number of gradient steps

    Returns:
        (eps_hit, x_adv_best, clean_pred, adv_pred_final)
        eps_hit: smallest L_inf norm at which prediction flips (None if never)
        x_adv_best: adversarial example at eps_hit (or last x if None)
        clean_pred: original prediction
        adv_pred_final: prediction at x_adv_best
    """
    model.eval()

    # original sample with batch dim
    x_orig = x.unsqueeze(0).to(device)   # (1, N, 3)
    y = y.to(device)

    # prediction on the clean sample
    with torch.no_grad():
        logits = model(x_orig, save_layers=False)
        clean_pred = logits.argmax(1).item()

    # If already misclassified, distance ~ 0
    if clean_pred != y.item():
        return 0.0, x_orig.detach().cpu(), clean_pred, clean_pred

    # initialize adversarial example
    x_adv = x_orig.clone().detach()
    x_adv.requires_grad_(True)

    # step size in L_inf radius per iteration
    alpha = eps_max / n_steps

    eps_hit = None
    adv_pred_final = clean_pred

    for _ in range(1, n_steps + 1):
        # forward + loss
        logits = model(x_adv, save_layers=False)
        loss = F.cross_entropy(logits, y.unsqueeze(0))

        # backward
        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss.backward()

        # gradient step
        grad_sign = x_adv.grad.data.sign()
        x_adv = x_adv + alpha * grad_sign

        # project back into L_inf ball of radius eps_max around x_orig
        eta = torch.clamp(x_adv - x_orig, min=-eps_max, max=eps_max)
        x_adv = (x_orig + eta).detach()
        x_adv.requires_grad_(True)

        # check prediction after this step
        with torch.no_grad():
            logits_adv = model(x_adv, save_layers=False)
            adv_pred = logits_adv.argmax(1).item()

        # current epsilon = actual L_inf norm of perturbation
        delta = x_adv - x_orig
        eps_now = float(delta.abs().max().item())

        # first time the prediction changes → we record eps_now and stop
        if adv_pred != y.item():
            eps_hit = eps_now
            adv_pred_final = adv_pred
            break

    if eps_hit is None:
        # never flipped within eps_max
        adv_pred_final = clean_pred if 'adv_pred' not in locals() else adv_pred

    return eps_hit, x_adv.detach().cpu(), clean_pred, adv_pred_final


def find_one_correct_sample_of_class(model, loader, device, class_id: int):
    """
    Return one point cloud of a given class that the model classifies correctly.
    """
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, save_layers=False)
            preds = logits.argmax(1)
            for i in range(x.size(0)):
                if int(y[i]) == class_id and int(preds[i]) == class_id:
                    # Return CPU tensors without batch dimension
                    return x[i].cpu(), y[i].cpu()
    return None, None



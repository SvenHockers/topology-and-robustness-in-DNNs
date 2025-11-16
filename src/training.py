import torch
import torch.nn.functional as F


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x, save_layers=False)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        correct += int((preds.argmax(1) == y).sum().item())
        total += int(x.size(0))
    return total_loss / max(total, 1), correct / max(total, 1)


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



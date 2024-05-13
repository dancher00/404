import torch
from compute_metrics import compute_metrics

def validating(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_rmse = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            mse, rmse = compute_metrics(outputs, targets)
            total_rmse += rmse * inputs.size(0)
            total_samples += inputs.size(0)

    avg_loss = running_loss / total_samples
    avg_rmse = total_rmse / total_samples
    return avg_loss, avg_rmse

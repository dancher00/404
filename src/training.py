import torch
from torch import nn

def compute_metrics(outputs, targets):
    mse = nn.MSELoss()(outputs, targets)
    rmse = torch.sqrt(mse)
    return mse.item(), rmse.item()

def training(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_rmse = 0.0
    total_samples = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        mse, rmse = compute_metrics(outputs, targets)
        
        total_rmse += rmse * inputs.size(0)
        total_samples += inputs.size(0)

    avg_loss = running_loss / total_samples
    avg_rmse = total_rmse / total_samples

    return avg_loss, avg_rmse
